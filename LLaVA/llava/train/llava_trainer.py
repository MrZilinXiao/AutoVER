import os
import torch
import json
import numpy as np
from collections import defaultdict
import warnings

import torch.nn as nn
from torch.utils.data import Sampler, ConcatDataset

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)

from typing import Iterator, List, Optional, Sized, Dict, Union, Any
from common_utils.trainer_utils import hack_callbacks_and_replace
from common_utils.sampling_utils import (
    get_one_class_per_batch_indices,
    get_hard_negative_per_batch_indices, 
    get_one_class_per_batch_indices_v2,
)

# for subclass imports
from transformers.utils import (
    is_peft_available, 
    is_apex_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
if is_peft_available():
    from peft import PeftModel

if is_apex_available():
    from apex import amp

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # sort by length in a big-to-small order
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    

class HardNegativeSampler(Sampler):
    """
    Sampler that organizes mega batch in hard-negative sampling way.
    """
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        labels: Optional[List[int]],
        hard_negative_groups: Optional[Dict[int, List[int]]],
        hard_negative_ratio: float,
        generator=None,
    ) -> None:
        self.batch_size = batch_size
        self.world_size = world_size
        self.labels = labels
        self.generator = generator

        # for computing random.choices weights
        self.label_to_count = defaultdict(int)
        for label in self.labels:
            self.label_to_count[label] += 1
        self.total_len = len(self.labels)
        
        self.hard_negative_groups = hard_negative_groups
        self.hard_negative_ratio = hard_negative_ratio

    def __len__(self):
        return len(self.labels)
    
    def __iter__(self) -> Iterator:
        indices = get_hard_negative_per_batch_indices(self.labels, self.batch_size, self.world_size, generator=self.generator, 
                                                  label_to_count=self.label_to_count, total_len=self.total_len, 
                                                  hard_negative_groups=self.hard_negative_groups, hard_negative_ratio=self.hard_negative_ratio)
        print(f"Previous 100 indices in sampler: {indices[:100]}")
        return iter(indices)

class OneClassPerBatchSampler(Sampler):  # Sampler are running on each main worker, but not on dataloader worker.
    r"""
    Sampler that ensure only one entity class per batch, allowing contrastive loss to learn; 
    """
    def __init__(
        self, 
        batch_size: int,
        world_size: int,
        labels: Optional[List[int]],
        generator=None,
    ) -> None:
        self.batch_size = batch_size
        self.world_size = world_size
        self.labels = labels  # len(self.labels) == size of train_set
        self.generator = generator

        # for computing random.choices weights
        self.label_to_count = defaultdict(int)
        for label in self.labels:
            self.label_to_count[label] += 1
        self.total_len = len(self.labels)
        

    def __len__(self):
        return len(self.labels)
    
    def __iter__(self) -> Iterator:
        # indices = get_one_class_per_batch_indices(self.labels, self.batch_size, self.world_size, generator=self.generator, 
        indices = get_one_class_per_batch_indices_v2(self.labels, self.batch_size, self.world_size, generator=self.generator, 
                                                  label_to_count=self.label_to_count, total_len=self.total_len)
        print(f"Previous 100 indices in sampler: {indices[:100]}")
        return iter(indices)

class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.losses_for_record = dict()  # FUCK! inner_training_loop will overwrite this
        self.callback_handler = hack_callbacks_and_replace(self.callback_handler)

    # Sorry: https://github.com/microsoft/DeepSpeed/issues/3310; Modifying grads in-place in ZeRO is not supported yet. 
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     if not self.args.tune_mm_mlp_adapter:
    #         return super().training_step(model, inputs)
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     # if is_sagemaker_mp_enabled():   # we do not use this branch
    #     #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     #     return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.do_grad_scaling:
    #         self.scaler.scale(loss).backward()
    #     elif self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss)

    #     # clear grads for non-ret token params; even before grad_accumulation, all though this will waste some time
    #     # hardcoded to save time
    #     ret_token_id = 32000
    #     for p in unwrap_model(model).get_input_embeddings().parameters():
    #         assert p.grad.shape[0] == 32064
    #         zero_grad_masks = torch.arange(p.grad.shape[0]) != ret_token_id
    #         p.grad[zero_grad_masks, :] = 0.
    #     for p in unwrap_model(model).lm_head.parameters():
    #         assert p.grad.shape[0] == 32064
    #         zero_grad_masks = torch.arange(p.grad.shape[0]) != ret_token_id
    #         p.grad[zero_grad_masks, :] = 0.
    #     warnings.warn("Hardcoded zero_grad_masks for ret_token_id=32000; clear grads for non-ret token params")

    #     return loss.detach() / self.args.gradient_accumulation_steps


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        overwrite the trainer to collect
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # collect all keys end with _loss; divided by accumulation steps; gather them, mean them then log as scalars
            if self.state.global_step % self.args.logging_steps == 0:
                for k, v in outputs.items():
                    if k == 'loss':
                        k = 'debug_loss'  # should be the same with `loss`
                    if k.endswith('loss'):
                        # any log call backs will retrieve the loss from state.losses_for_record
                        self.state.losses_for_record[k] = self._nested_gather(v).mean().item()
                    if k.endswith('acc'):  # acc does not need gather; they are scalars
                        self.state.losses_for_record[k] = v.item()

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        
        elif self.args.hard_negative_group_path is not None:
            print("Using hard negative sampler! Usually do not use it if retrieval is not enabled!")
            if isinstance(self.train_dataset, ConcatDataset):
                labels = []
                for dataset in self.train_dataset.datasets:
                    labels.extend(dataset.entity_labels)
            else:
                labels = self.train_dataset.entity_labels

            with open(self.args.hard_negative_group_path, 'r') as f:
                hard_negative_groups = json.load(f)
            
            # have to go through Q-striping for every qid
            hard_negative_groups = {int(k[1:]): [int(vv[1:]) for vv in v] for k, v in hard_negative_groups.items()}

            return HardNegativeSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                labels=labels,
                hard_negative_groups=hard_negative_groups,
                hard_negative_ratio=self.args.hard_negative_ratio,
            )
        
        elif self.args.one_class_per_batch:
            print("Using one class per batch sampler! Usually do not use it if retrieval is not enabled!")
            if isinstance(self.train_dataset, ConcatDataset):
                labels = []
                for dataset in self.train_dataset.datasets:
                    labels.extend(dataset.entity_labels)
            else:
                labels = self.train_dataset.entity_labels

            return OneClassPerBatchSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                labels=labels,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None or self.args.ret_projector_lr is not None or self.args.fusion_encoder_lr is not None:
                self.args.mm_projector_lr = self.args.learning_rate if self.args.mm_projector_lr is None else self.args.mm_projector_lr
                self.args.ret_projector_lr = self.args.learning_rate if self.args.ret_projector_lr is None else self.args.ret_projector_lr
                self.args.fusion_encoder_lr = self.args.learning_rate if self.args.fusion_encoder_lr is None else self.args.fusion_encoder_lr

                mm_projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                ret_projector_parameters = [name for name, _ in opt_model.named_parameters() if "ret_token_projector" in name]
                fusion_encoder_parameters = [name for name, _ in opt_model.named_parameters() if "fusion_encoder" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [  # normal decay params
                            p for n, p in opt_model.named_parameters() if (
                                n in decay_parameters and 
                                n not in mm_projector_parameters and 
                                n not in ret_projector_parameters and 
                                n not in fusion_encoder_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [  # normal no decay params
                            p for n, p in opt_model.named_parameters() if (
                                n not in decay_parameters and 
                                n not in mm_projector_parameters and 
                                n not in ret_projector_parameters and 
                                n not in fusion_encoder_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    # Existing mm_projector layers
                    {
                        "params": [  # decay mm_projector params
                            p for n, p in opt_model.named_parameters() if (
                                n in decay_parameters and 
                                n in mm_projector_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [  # no decay mm_projector params
                            p for n, p in opt_model.named_parameters() if (
                                n not in decay_parameters and 
                                n in mm_projector_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    # New ret_token_projector layers
                    {
                        "params": [  # decay ret_projector params
                            p for n, p in opt_model.named_parameters() if (
                                n in decay_parameters and 
                                n in ret_projector_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.ret_projector_lr,
                    },
                    {
                        "params": [  # no decay ret_projector params
                            p for n, p in opt_model.named_parameters() if (
                                n not in decay_parameters and 
                                n in ret_projector_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.ret_projector_lr,
                    },
                    # New fusion_encoder layers
                    {
                        "params": [  # decay fusion_encoder params
                            p for n, p in opt_model.named_parameters() if (
                                n in decay_parameters and 
                                n in fusion_encoder_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.fusion_encoder_lr,
                    },
                    {
                        "params": [  # no decay fusion_encoder params
                            p for n, p in opt_model.named_parameters() if (
                                n not in decay_parameters and 
                                n in fusion_encoder_parameters and 
                                p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.fusion_encoder_lr,
                    },
                ]
                # optimizer_grouped_parameters = [
                #     {
                #         "params": [
                #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in mm_projector_parameters and p.requires_grad)
                #         ],
                #         "weight_decay": self.args.weight_decay,
                #     },
                #     {
                #         "params": [
                #             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in mm_projector_parameters and p.requires_grad)
                #         ],
                #         "weight_decay": 0.0,
                #     },
                #     {
                #         "params": [
                #             p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in mm_projector_parameters and p.requires_grad)
                #         ],
                #         "weight_decay": self.args.weight_decay,
                #         "lr": self.args.mm_projector_lr,
                #     },
                #     {
                #         "params": [
                #             p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in mm_projector_parameters and p.requires_grad)
                #         ],
                #         "weight_decay": 0.0,
                #         "lr": self.args.mm_projector_lr,
                #     },
                # ]
            else:  # normal branch, no weight_decay for layer_norm
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)