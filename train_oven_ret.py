# Simplified version of training LLaVA variants on OVEN dataset; ignore features we won't use. 
# Offline evaluation is needed with constrained decoding.
import sys
sub_paths = [
    "./LLaVA/"
]
for sub_path in sub_paths:
    sys.path.insert(0, sub_path)  # hack to allow import in the sub directory

import torch
from dataclasses import dataclass, field
from typing import Optional, List
import transformers
import logging
import os
import pathlib
from LLaVA.llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM, LlavaLlamaForCausalLMWithRet, LlavaLlamaForEntityClassification
)
from LLaVA.llava.train.llava_trainer import LLaVATrainer
from common_utils.oven_data import build_oven_data_modules
from common_utils.oven_data_cls import build_oven_data_modules_cls
from datetime import datetime


local_rank = None

def rank0_print(*args):  # debug usage only
    if local_rank == 0:
        print(*args)

@dataclass
class DataArguments:
    is_multimodal: bool = False  # forced multimodal pair
    # below for oven dataset
    train_jsonl_path: str = field(default=None)
    multi_val_jsonl_path: List[str] = field(default=None)   # do not touch, for merge-eval use only
    train_img_shards_path: str = field(default=None)
    train_qid_to_entity_path: str = field(default="trie_bins/wiki_qid_to_entity_indata.json")
    # summary is disabled by default
    use_summary: bool = field(default=False)
    qid_to_summary_path: str = field(default=None)
    max_summary_tokens: int = field(default=128)

    query_jsonl_path: List[str] = field(default=None)  # if enabled, will mix with train_jsonl_path with query_sample_ratio
    query_sample_ratio: List[int] = field(default=None)
    
    entity_img_path: str = field(default=None)
    # above for oven dataset
    image_aspect_ratio: str = 'pad'
    image_grid_pinpoints: Optional[str] = field(default=None)
    

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-13b-v1.5")  # not instructed model (just complete stage1); 
    # meet our requirements
    version: Optional[str] = field(default="v1")  # finetune use 'v1', however we don't really care as it controls
    # prompt template we don't use
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)  # if true, only train mm_projector for feature alignment
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')  # default is a linear
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)  # in all training deault to False
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # added for retrieval task
    retrieval: bool = field(default=False)
    entity_encoder: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    entity_encoder_fusion: Optional[str] = field(default="mlp2x_gelu")  # trm, film, '^mlp(\d+)x_gelu$'
    retrieval_shared_dim: Optional[int] = field(default=512)  # clip_text & clip_vision gets a 768 hidden_dim
    # pretrain_retrieval_modules: Optional[str] = field(default=None)  # load ret_token_projector & entity_encoder.fusion_encoder
    retrieval_loss_alpha: Optional[float] = field(default=1.)  # let's get a middle checkpoint & see acc @ 1,5,10,50,100
    pad_to_multiple_of: Optional[int] = field(default=64)  # for comp > 7.5 device
    gather_contrastive: Optional[bool] = field(default=True)  # gather contrastive loss
    load_retrieval_modules: Optional[bool] = field(default=False)  # True when continue training / eval on frozen model + retrieval modules
    ret_to_entity_only: Optional[bool] = field(default=True)
    unfreeze_entity_clip: Optional[bool] = field(default=False)
    # above added for retrieval task
    # added for cls task
    entity_classification: bool = field(default=False)
    qid_to_cls_path: Optional[str] = field(default="trie_bins/qid_to_cls.json")



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)  # freeze the visual projection head?
    mpt_attn_impl: Optional[str] = field(default="triton")  # only when mpt in model_name; don't touch
    model_max_length: int = field(
        default=512,  # 512 is enough if we do not concat samples to increase efficency
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # below only useful when bits = 4 / 8
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    # above only useful when bits = 4 / 8
    bits: int = field(
        default=16,  # 16 bits training?
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    # added for retrieval task
    ret_projector_lr: Optional[float] = field(default=None)
    fusion_encoder_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    one_class_per_batch: bool = field(default=False)
    hard_negative_group_path: str = field(default=None)
    hard_negative_ratio: float = field(default=0.5)
    # above added for retrieval task
    
    def __post_init__(self):
        super().__post_init__()
        if self.report_to == "wandb":
            # prepend mmddyyyy format datetime before the run_name
            self.run_name = datetime.now().strftime("%m%d%Y") + "_" + self.run_name


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    # TODO: add save retrieval modules only when tune_retrieval_modules is True
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'embed_tokens', 'lm_head', 'fusion_encoder', 'ret_token_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if model_args.retrieval:
            # model_name_or_path = model_args.model_name_or_path if model_args.extend_model_name_or_path is None else model_args.extend_model_name_or_path
            model = LlavaLlamaForCausalLMWithRet.from_pretrained(  # retrieval branch
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.entity_classification:
            model = LlavaLlamaForEntityClassification.from_pretrained(  # cls branch
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            model.config.num_entities = 20549
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(  # this branch; mpt is for diff impl of attn
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:  # training LM or not; this may change when resizing token embedding
        model.model.requires_grad_(False)

    # if training_args.bits in [4, 8]:
    #     from peft import prepare_model_for_kbit_training
    #     model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    data_args.retrieval = model_args.retrieval
    data_args.entity_image_processor = None
    data_args.entity_tokenizer = None
    data_args.qid_to_cls_path = model_args.qid_to_cls_path

    if model_args.retrieval:
        model.get_model().initialize_retriever_modules(
            model_args=model_args,
        )
        model.get_model().move_retrieval_to_device(dtype=compute_dtype, device=training_args.device)
        data_args.entity_image_processor = model.get_model().entity_encoder.entity_image_processor
        data_args.entity_tokenizer = model.get_model().entity_encoder.entity_text_tokenizer
        # ABANDON: Deepspeed does not allow grad control
        # finally, copy args from model_args to data_args for dataset init
        data_args.entity_image_processor = model.get_model().entity_encoder.entity_image_processor
        data_args.entity_tokenizer = model.get_model().entity_encoder.entity_text_tokenizer
        # resize_token_embeddings will restore requires_grad in lm_head & embed_tokens
        model.config.gather_contrastive = model_args.gather_contrastive

    elif model_args.entity_classification:
        model.get_model().initialize_classifer_modules(model_args)
        model.get_entity_classifier().to(dtype=compute_dtype, device=training_args.device)

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(  # should we also lazy load the vision tower?
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)  # freeze entire model;
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if model_args.retrieval:  # unfreeze fusion_encoder / ret_token_projector / lm_head / embed_tokens
                for p in model.get_model().get_entity_encoder().fusion_encoder.parameters():
                    p.requires_grad = True
                for p in model.get_model().get_ret_token_projector().parameters():
                    p.requires_grad = True
                for p in model.lm_head.parameters():
                    p.requires_grad = True
                for p in model.get_input_embeddings().parameters():
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # in notebook this seems to be moved manually
        # model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if model_args.entity_classification:
        data_module = build_oven_data_modules_cls(tokenizer=tokenizer,
                                         data_args=data_args,
                                         is_training=True,)
    else:
        data_module = build_oven_data_modules(tokenizer=tokenizer,
                                            data_args=data_args,
                                            is_training=True,)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
