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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .entity_encoder.builder import build_entity_encoder

from llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, DEFAULT_RET_TOKEN
)
import transformers

class EntityClassificationMetaModel:
    def __init__(self, config) -> None:
        super(EntityClassificationMetaModel, self).__init__(config)
        self.config = config

        if hasattr(config, "num_entities"):
            self.entity_classifier = nn.Linear(config.hidden_size, config.num_entities)

    def get_entity_classifier(self):
        return getattr(self, 'entity_classifier', None)

    def initialize_classifer_modules(self, model_args, force_init=False):
        """
        Copying entity classification related args to config; useful when train from non-cls model
        """
        if not hasattr(self.config, 'num_entities'):
            self.config.num_entities = model_args.num_entities
        # if hasattr(self.config, 'entity_list'):
        #     self.config.entity_list = model_args.entity_list   # corresponding to num_entities
        
        if force_init or (getattr(self, 'entity_classifier', None) is None and hasattr(self.config, 'num_entities')):
            self.entity_classifier = nn.Linear(self.config.hidden_size, self.config.num_entities)
            print("EntityClassificationMetaModel: entity_classifier initialized from random weights! This is NOT expected if you are doing eval.")

        else:
            print("EntityClassificationMetaModel: entity_classifier already initialized!")

        # we usually does not train the entity classifier alone

class RetrieverMetaModel:  # Done
    # only handles retrieval model init: leave the loss compute in the `ForCasualLM` class
    # leave the `prepare_inputs_labels_for_multimodal` for ForCasualLM class as well
    # we should only allow fusion layer to be trained.
    def __init__(self, config) -> None:
        super(RetrieverMetaModel, self).__init__(config)
        self.config = config
        if hasattr(config, "retrieval") and config.retrieval:  # in from_scratch training 
            # (from base model, e.g. vicuna), this will not be load;
            # we have to init them in `initialize_retriever_modules`
            self.entity_encoder = build_entity_encoder(config)
            self.ret_token_projector = nn.Sequential(  # from llava hidden_size to retrieval_shared_dim
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Linear(self.config.hidden_size, self.config.retrieval_shared_dim),
            )

    def get_entity_encoder(self):
        return getattr(self, 'entity_encoder', None)
    
    def get_ret_token_projector(self):
        return getattr(self, 'ret_token_projector', None)
    
    def move_retrieval_to_device(self, device, dtype):
        if hasattr(self, 'entity_encoder'):
            self.entity_encoder.to(device=device, dtype=dtype)
        if hasattr(self, 'ret_token_projector'):
            self.ret_token_projector.to(device=device, dtype=dtype)

    # the reason to `initialize_retriever_modules` is sometimes we load 
    # the ret_token_projector & entity_encoder.fusion_encoder for base model
    def initialize_retriever_modules(self, model_args, force_init=False):
        # copy retriever-related args to config
        # retrieval: bool = field(default=False)
        # entity_encoder: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
        # entity_encoder_fusion: Optional[str] = field(default="mlp2x_gelu")  # trm, film, '^mlp(\d+)x_gelu$'
        # retrieval_shared_dim: Optional[int] = field(default=512)  # clip_text & clip_vision gets a 768 hidden_dim
        # pretrain_retrieval_modules: Optional[str] = field(default=None)  # load ret_token_projector & entity_encoder.fusion_encoder
        # Update on 11/10: if self.config already has these attributes, we will not override them
        if not hasattr(self.config, 'retrieval'):
            self.config.retrieval = model_args.retrieval
        if not hasattr(self.config, 'entity_encoder'):
            self.config.entity_encoder = model_args.entity_encoder
        if not hasattr(self.config, 'entity_encoder_fusion'):
            self.config.entity_encoder_fusion = model_args.entity_encoder_fusion
        if not hasattr(self.config, 'retrieval_shared_dim'):
            self.config.retrieval_shared_dim = model_args.retrieval_shared_dim
        if not hasattr(self.config, 'pretrain_retrieval_modules'):
            self.config.retrieval_loss_alpha = model_args.retrieval_loss_alpha
        if not hasattr(self.config, 'ret_to_entity_only'):
            self.config.ret_to_entity_only = model_args.ret_to_entity_only
        if not hasattr(self.config, 'unfreeze_entity_clip'):
            self.config.unfreeze_entity_clip = model_args.unfreeze_entity_clip

        if force_init or (getattr(self, 'entity_encoder', None) is None and self.config.retrieval):
            self.entity_encoder = build_entity_encoder(self.config)
            self.ret_token_projector = nn.Sequential(  # from llava hidden_size to retrieval_shared_dim
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Linear(self.config.hidden_size, self.config.retrieval_shared_dim),
            )
            print("RetrieverMetaModel: entity_encoder & ret_projector initialized from random weights! This is NOT expected if you are doing eval.")
        else:
            print("RetrieverMetaModel: entity_encoder & ret_projector already initialized!")

        # load weights from pretrain_retrieval_modules
        pretrain_retrieval_modules = getattr(model_args, 'pretrain_mm_mlp_adapter', None)
        if model_args.load_retrieval_modules and pretrain_retrieval_modules is not None:
            # when saving, these two layers need special treatment
            retrieval_weights = torch.load(pretrain_retrieval_modules, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.ret_token_projector.load_state_dict(get_w(retrieval_weights, 'ret_token_projector'))
            self.entity_encoder.fusion_encoder.load_state_dict(get_w(retrieval_weights, 'entity_encoder.fusion_encoder'))
            print(f"Loaded pretrained ret_token_projector & entity_encoder.fusion_encoder from {pretrain_retrieval_modules}!")

            # retrieval_weights may have lm_head & embed_tokens weights as well; we will help to handle it
            if any('lm_head' in k for k in retrieval_weights.keys()):
                self.get_model().lm_head.load_state_dict(get_w(retrieval_weights, 'lm_head'))
                print(f"Loaded pretrained lm_head from {pretrain_retrieval_modules}!")

            if any('embed_tokens' in k for k in retrieval_weights.keys()):
                self.get_model().embed_tokens.load_state_dict(get_w(retrieval_weights, 'embed_tokens'))
                print(f"Loaded pretrained embed_tokens from {pretrain_retrieval_modules}!")


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    # init vision tower will directly append model_args to pretrained model config
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()   # load here when delay_load=True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:  # from CLIP space to LLaVA space
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print(f"Loaded pretrained mm_projector from {pretrain_mm_mlp_adapter}!")

class LlavaMetaForCausalLMWithRet(ABC):  # abstract class: DO NOT init
    @abstractmethod
    def get_model(self):
        pass

    def get_ret_token_projector(self):
        return self.get_model().get_ret_token_projector()  # in `RetrieverMetaModel`
    
    def get_entity_encoder(self):
        return self.get_model().get_entity_encoder()  # in `RetrieverMetaModel`
    
    def encode_entities(self, entity_images, entity_token_ids, entity_attention_mask):
        entity_encoder = self.get_entity_encoder()
        assert entity_encoder is not None, "entity_encoder is None!"
        return entity_encoder(entity_images, entity_token_ids, entity_attention_mask)
    
    # WARNING: do not call this in deepspeed ZeRO3
    def initialize_retrieval_tokenizer(self, model_args, tokenizer: transformers.PreTrainedTokenizer):
        # call this in train in parallel with `initialize_vision_tokenizer`
        # this will also expand the vocab size as well as unfreezing the embedding layer
        # num_added_tokens = tokenizer.add_tokens(DEFAULT_RET_TOKEN)
        # ret_token_id = tokenizer(DEFAULT_RET_TOKEN, add_special_tokens=False)['input_ids'][-1]
        special_tokens_dict = {'additional_special_tokens': ['<ret>', '<ent_cls>']}
        num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        ret_token_id = tokenizer(DEFAULT_RET_TOKEN)['input_ids'][-1]
        
        print(f"Added {num_added_tokens} tokens to tokenizer: {DEFAULT_RET_TOKEN} with token_id: {ret_token_id}")
        self.config.ret_token_id = model_args.ret_token_id = ret_token_id

        pad_to_multiple_of = getattr(model_args, 'pad_to_multiple_of', 64)
        self.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)  # this controls lm_head + embed_tokens
        self.config.vocab_size = model_args.vocab_size = self.get_model().get_input_embeddings().weight.shape[0]

        return num_added_tokens


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    # does the vision tower get trained during finetuning? No!
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # print(f"image_features.dtype: {image_features.dtype}")
        # print(f"image_features.shape: {image_features.shape}; mm_projector: {self.get_model().mm_projector}")
        image_features = self.get_model().mm_projector(image_features)
        # mm_hidden_size -> hidden_size
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, 
    ):
        vision_tower = self.get_vision_tower()
        # if no vision tower, just return or do some attention mask padding
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:  # multiple imgs of list
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):  # for each sample
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:  
                # Zilin: we don't go this branch; we ensure LM will receive an image
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]  # select image features by order
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # find where the image token lies
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # starting from empty image features
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                # Zilin: don't go this branch; we do not use im_start/end
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:  # Zilin: we add image_features right after previous text tokens
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                # just a <img> or <img_st> + <img> + <img_ed>
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                # Zilin: double-check to detach embed_tokens
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):  
            # possible padding for input_embeds
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # if keep_input_ids:
        #     return input_ids, attention_mask, past_key_values, new_input_embeds, new_labels
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # our impl does not involving adding image_patch, st, ed tokens in vocab
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
