#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import (
    LlavaMetaModel, LlavaMetaForCausalLM, 
    RetrieverMetaModel, LlavaMetaForCausalLMWithRet, 
    EntityClassificationMetaModel
)
from dataclasses import dataclass
import torch.distributed as dist
from common_utils.dist_utils import (
    all_gather_with_grad as all_gather_with_grad_lavis, 
    all_gather_with_grad_torch
)
from common_utils.loss_utils import contrastive_loss, contrastive_acc
from llava.constants import DEFAULT_ENT_CLS_TOKEN_INDEX, IGNORE_INDEX
# official grad-version of all_gather might help


@dataclass
class CausalLMOutputWithPastWithRet(CausalLMOutputWithPast): 
    # these losses are for gathering and stats only
    ce_loss: Optional[torch.FloatTensor] = None
    ret_loss: Optional[torch.FloatTensor] = None  # ret2ent
    ent_loss: Optional[torch.FloatTensor] = None  # ent2ret
    retrieval_loss: Optional[torch.FloatTensor] = None
    ret_1_acc: Optional[float] = None
    ret_10_acc: Optional[float] = None
    ret_100_acc: Optional[float] = None
    ent_1_acc: Optional[float] = None
    ent_10_acc: Optional[float] = None
    ent_100_acc: Optional[float] = None

@dataclass
class CausalLMOutputWithPastWithCLS(CausalLMOutputWithPastWithRet):
    # single-objective, so we just use loss from parent
    # but we want to output prediction for entity classification
    pred: Optional[torch.LongTensor] = None  # N

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaLlamaModelWithRet(LlavaMetaModel, RetrieverMetaModel, LlamaModel):  # add RET meta to `LlavaLlamaModelWithRet`
    # load a lazy vision tower, a retriever and a llama LM in order.
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModelWithRet, self).__init__(config)

class LlavaLlamaModelWithCls(LlavaMetaModel, EntityClassificationMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModelWithCls, self).__init__(config)

# On Nov. 9, implement Vicente's idea of classifier-based VQA using the same backbone
# we insert a new token <ent> and attach its hidden states to a classifier on all known (seen + unseen) entities
class LlavaLlamaForEntityClassification(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    """
    Notice: we do not subclass from LlavaMetaForEntityClassification to avoid multi-level abstraction
    This requires us to have a model.config with `num_entities` and `entity_list` ahead of time
    """

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)  
        self.model = LlavaLlamaModelWithCls(config)

        # does not need a lm_head anymore
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_entity_classifier(self):
        return self.model.get_entity_classifier()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        # added entity_cls_labels for entity classification
        # entity_cls_pos: Optional[torch.LongTensor] = None,
        # N; indicating the position of <ent_cls> token: run a notebook to assert this with embed_tokens[32001]
        # entity_cls_labels: Optional[torch.LongTensor] = None,   
        # N, num_entities
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # assert input_ids with <ent_cls>
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)

        # we need the labels to know where the cls token in each sample sits
        assert labels is not None, "labels is None; we need labels to know where the <ent_cls> token is"
        loss_fct = CrossEntropyLoss()
        cls_batch, cls_pos = torch.where(labels != IGNORE_INDEX)
        cls_labels = labels[cls_batch, cls_pos]
        cls_hidden_states = hidden_states[cls_batch, cls_pos]
        cls_logits = self.get_entity_classifier()(cls_hidden_states)

        loss = loss_fct(cls_logits, cls_labels)

        # for quick evaluation, we also output current prediction
        pred = torch.argmax(cls_logits, dim=-1)

        return CausalLMOutputWithPastWithCLS(
            loss=loss,
            logits=cls_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pred=pred
        )



# the final model we are run .from_pretrained() on
# should implement: `forward` (with loss), `prepare_inputs_for_generation`
class LlavaLlamaForCausalLMWithRet(LlamaForCausalLM, LlavaMetaForCausalLM, LlavaMetaForCausalLMWithRet):
    # LlavaMetaForCausalLM provides impls for left-tower vision model; 
    # LlavaMetaForCausalLMWithRet provides impls for right-tower entity model
    config_class = LlavaConfig

    def __init__(self, config):  # from_pretrained will load weights; 
        # how do we pass custom args into? just copy model_args in MetaClass
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModelWithRet(config)
        # llava lm_head is not tied with embed_tokens;
        # `resize_token_embeddings`` automatically resizes the lm_head when 
        # resizing for `LlavaLlamaForCausalLM`
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        # added for entity input; 
        entity_images: Optional[torch.FloatTensor] = None,
        # remember to update the keys from `entity_tokenizer`
        entity_token_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.LongTensor] = None,  # only with labels, we compute loss
        # you may need entity_id (Long) to select hard negatives
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # FUCK: <ret> position will change in input_ids when trying to insert image...
        outputs = self.model(
            input_ids=input_ids,  
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # print(f'processed labels: {labels.tolist()}')

        hidden_states = outputs[0]  # N, L, 4096
        # assert hidden_states.shape[1] == inputs_embeds.shape[1], f"hidden_states & input_embeds shape mismatch!"
        logits = self.lm_head(hidden_states)
        # ce loss first
        loss = None
        ce_loss = None
        ret_loss = None
        ent_loss = None
        retrieval_loss = None
        ret_acc_1, ret_acc_10, ret_acc_100, ent_acc_1, ent_acc_10, ent_acc_100 = [None] * 6
        do_ret_this_batch = True

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)

        # above copied from LlavaLlamaForCausalLM; now we need to collect entity features
        if entity_labels is not None:
            entity_features = self.get_entity_encoder()(entity_images, 
                                                        entity_token_ids, 
                                                        entity_attention_mask)  # N, retrieval_dim
            # now get hidden states from <ret> token id, which is updated in self.config.ret_token_id 
            ret_token_id = self.config.ret_token_id
            assert ret_token_id is not None, "ret_token_id is not set in config"
            ret_batch_ids, ret_pos_ids = torch.where(labels == ret_token_id)  # ensure <ret> is not IGNORED!
            # input_embeds[ret_pos_ids] should be the same with self.get_model().get_input_embeddings()[32000]
            # print(inputs_embeds[ret_batch_ids, ret_pos_ids])  # compare passed...
            # print(hidden_states[ret_batch_ids, ret_pos_ids])
            # print(self.get_model().get_input_embeddings().weight[32000])

            # each and every sample has ONLY one <ret> token
            assert ret_batch_ids.tolist() == list(range(len(ret_batch_ids))), f"ret_batch_ids: {ret_batch_ids}; we only allow one <ret> token per sample"
            ret_token_features = hidden_states[ret_batch_ids, ret_pos_ids]  # N, 4096
            ret_token_features = self.get_ret_token_projector()(ret_token_features)  # N, retrieval_dim
            # TODO: Any other dist implementation here?
            if dist.is_initialized() and self.config.gather_contrastive:  # ret_loss dropped but slowly...
                entity_features = all_gather_with_grad_lavis(entity_features)
                ret_token_features = all_gather_with_grad_lavis(ret_token_features)
                entity_labels = all_gather_with_grad_lavis(entity_labels)  # [431223, 2131, 78923, 2131]

            # fromage version of all_gather; please attach deepspeed-compatible debugger here.
            # if dist.is_initialized() and self.config.gather_contrastive:
            #     all_entity_features = [torch.zeros_like(entity_features) for _ in range(dist.get_world_size())]
            #     all_ret_token_features = [torch.zeros_like(ret_token_features) for _ in range(dist.get_world_size())]
            #     dist.all_gather(all_entity_features, entity_features)
            #     dist.all_gather(all_ret_token_features, ret_token_features)
            #     all_entity_features[dist.get_rank()] = entity_features
            #     all_ret_token_features[dist.get_rank()] = ret_token_features
            #     entity_features = torch.cat(all_entity_features)
            #     ret_token_features = torch.cat(all_ret_token_features)
            # reorg entity_labels for in-batch pos-neg learning  [431223, 2131, 78923, 2131] -> [0, 1, 2, 1]
            unique_labels = torch.unique(entity_labels)
            if not unique_labels.shape[0] == entity_labels.shape[0]:  # duplicated samples in ret
                do_ret_this_batch = False
                print(f"Skipping RET! Duplicated Entities in the global batch -- unique_labels size: {unique_labels.shape[0]}, entity_labels size: {entity_labels.shape[0]}")
            if do_ret_this_batch:
                # run norm before contrastive loss
                entity_features = F.normalize(entity_features, dim=-1)
                ret_token_features = F.normalize(ret_token_features, dim=-1)
                # ret2ent & ent2ret loss
                logits_per_ent = entity_features @ ret_token_features.t()    # entity to ret tokens
                logits_per_ret = logits_per_ent.t()   # ret tokens to entity
                # entity labels may be discrete or continuous
                if self.config.ret_to_entity_only:
                    ret_loss = contrastive_loss(logits_per_ret)
                    retrieval_loss = self.config.retrieval_loss_alpha * ret_loss
                    ret_acc_1, ret_acc_10 = contrastive_acc(logits_per_ret, topk=(1, 10, ))
                    # ent to ret acc & loss are not computed
                else:
                    ret_loss = contrastive_loss(logits_per_ret)
                    ent_loss = contrastive_loss(logits_per_ent)
                    retrieval_loss = self.config.retrieval_loss_alpha * (ret_loss + ent_loss) / 2.
                    # train constrastive acc is only for reference
                    ret_acc_1, ret_acc_10 = contrastive_acc(logits_per_ret, topk=(1, 10,))
                    ent_acc_1, ent_acc_10 = contrastive_acc(logits_per_ent, topk=(1, 10,))
            
        loss = ce_loss
        if loss is not None and retrieval_loss is not None:  # we do not allow optimize retrieval alone
            loss = loss + retrieval_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPastWithRet(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            ce_loss=ce_loss, 
            ret_loss=ret_loss,
            ent_loss=ent_loss,
            retrieval_loss=retrieval_loss, 
            ret_1_acc=ret_acc_1,
            ret_10_acc=ret_acc_10,
            ret_100_acc=ret_acc_100,
            ent_1_acc=ent_acc_1,
            ent_10_acc=ent_acc_10,
            ent_100_acc=ent_acc_100
        )


    # copied from LlavaLlamaForCausalLM; I don't see significant difference when involving RET
    def prepare_inputs_for_generation(  # pre-forward-hooked in GenerationMixin
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    # LlavaMetaForCausalLM provides impls for: 
    # def initialize_vision_tokenizer(self, model_args, tokenizer)  
    # manually called in train; expanding the tokenizer with image special tokens
    # def encode_images(self, images):  # get image features in selected ways
    # def get_vision_tower(self): just returning self.get_model().get_vision_tower()
    # def prepare_inputs_labels_for_multimodal(  # will be called in `forward`
    #     self, input_ids, attention_mask, past_key_values, labels, images
    # ):
        
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)  
        # will this first init a llama then a llava? this will be a problem
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(  # pre-forward-hooked in GenerationMixin
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMWithRet)