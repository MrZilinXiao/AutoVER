from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPVisionModel
from .wiki_encoder import EntityEncoder
from transformers import CLIPImageProcessor
from transformers import AutoTokenizer

def build_entity_encoder(entity_encoder_cfg, lazy_load=False):
    # add possible laod_from_llava weight cuz sometimes we unfreeze CLIP
    # Good news is that, it seems from_pretrained will automatically overwrite the weights
    # Wait for notebook testing...
    vision_encoder = CLIPVisionModel.from_pretrained(
        entity_encoder_cfg.entity_encoder
    )
    text_encoder = CLIPTextModel.from_pretrained(
        entity_encoder_cfg.entity_encoder
    )
    entity_image_processor = CLIPImageProcessor.from_pretrained(entity_encoder_cfg.entity_encoder)
    entity_text_tokenizer = AutoTokenizer.from_pretrained(entity_encoder_cfg.entity_encoder, use_fast=False)
    # copy necessary args to entity_encoder_cfg to init EntityEncoder
    entity_encoder_cfg.entity_vision_dim = vision_encoder.config.hidden_size
    entity_encoder_cfg.entity_text_dim = text_encoder.config.hidden_size

    return EntityEncoder(entity_encoder_cfg, 
                         vision_encoder, 
                         text_encoder, 
                         entity_image_processor, 
                         entity_text_tokenizer)
