# this should support both online encoder & offline encoder
import torch
import torch.nn as nn
import re
import transformers
from transformers import CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPVisionModel, CLIPTextModel
from .twoway_transformers import TwoWayTransformer, PositionEmbeddingRandom

class TwoLayerFusionTRM(nn.Module):
    IMAGE_PATCH_SIZE = 14
    IMAGE_EMBEDDING_SIZE = (24, 24)
    IMAGE_PATCH_NUM = 576

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # clip text patch & clip vision patch hidden size are not the same, need projector
        self.text_to_vision_projector = nn.Sequential(
            nn.Linear(config.entity_text_dim, config.entity_vision_dim),
            nn.GELU(),
            nn.Linear(config.entity_vision_dim, config.entity_vision_dim),
        )
        self.image_pe_layer = PositionEmbeddingRandom(config.entity_vision_dim // 2)
        self.transformer = TwoWayTransformer(
            depth=2, 
            embedding_dim=config.entity_vision_dim,  # 1024
            mlp_dim=2048,
            num_heads=8,
        )
        self.fusion_token = nn.Embedding(1, config.entity_vision_dim)
        self.fusion_to_shared_projector = nn.Sequential(
            nn.Linear(config.entity_vision_dim, config.retrieval_shared_dim),
            nn.GELU(),
            nn.Linear(config.retrieval_shared_dim, config.retrieval_shared_dim),
        )

    def get_dense_pe(self) -> torch.Tensor:
        return self.image_pe_layer(self.IMAGE_EMBEDDING_SIZE).unsqueeze(0)
    
    def forward(self, image_features, text_features):
        # image_features: N, 577 (cls + patch), config.entity_vision_dim; run layer selection outside
        # text_features: N, L, config.entity_text_dim
        # 1. project text to vision dim
        text_features = self.text_to_vision_projector(text_features)  # N, L, config.entity_vision_dim
        # concat self.fusion_token.weight (1, 1024) to the beginning of text_features, 
        # which will be projected to retrieval_shared_dim
        text_features = torch.cat([self.fusion_token.weight.unsqueeze(0).expand(text_features.size(0), -1, -1), 
                                   text_features], dim=1)  # N, 1 + L, config.entity_vision_dim
        image_features = image_features[:, 1:].view(-1, self.config.entity_vision_dim, *self.IMAGE_EMBEDDING_SIZE)
        # N, 576, config.entity_vision_dim
        image_pe = self.get_dense_pe().to(image_features.device)

        hidden_states, image_features = self.transformer(
            image_features, 
            image_pe, 
            text_features,
        )
        # hidden_states: N, 1 + L, config.entity_vision_dim
        fusion_token_out = hidden_states[:, 0, :]  # N, config.entity_vision_dim
        return self.fusion_to_shared_projector(fusion_token_out)  # N, retrieval_shared_dim


def build_fusion_projector(config):
    projector_type = getattr(config, 'entity_encoder_fusion', 'linear')

    if projector_type.startswith('trm'):
        return TwoLayerFusionTRM(config)

    projector_input_dim = config.entity_vision_dim + config.entity_text_dim

    if projector_type == 'linear':
        return nn.Linear(projector_input_dim, config.retrieval_shared_dim)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(projector_input_dim, config.retrieval_shared_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.retrieval_shared_dim, config.retrieval_shared_dim))
        return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')


class EntityEncoder(nn.Module):
    # only used in train; in eval, we will pre-cache the entity features
    def __init__(self, config, 
                 vision_encoder: CLIPVisionModel, 
                 text_encoder: CLIPTextModel, 
                 entity_image_processor, 
                 entity_text_tokenizer: transformers.PreTrainedTokenizer, 
                 ) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.entity_image_processor = entity_image_processor  # for external access
        self.entity_text_tokenizer = entity_text_tokenizer  # for external access

        unfreeze_entity_clip = getattr(config, 'unfreeze_entity_clip', False)   # backward compatibility
        if not unfreeze_entity_clip:
            print('Freezing entity encoder (vision & text)')
            self.vision_encoder.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
        else:
            print('Unfreezing entity encoder (vision & text)')

        self.fusion_encoder = build_fusion_projector(config)

    def forward(self, entity_images, entity_token_ids, entity_attention_mask):
        if self.config.entity_encoder_fusion == 'trm':  # trm has different fusion forward
            image_hidden_states = self.vision_encoder(entity_images, output_hidden_states=True).hidden_states
            image_features = image_hidden_states[-2]  # second last layer, N, 1 + 576, 1024
            text_features = self.text_encoder(entity_token_ids, entity_attention_mask)[0]  # last hidden layer, N, L, 768
            return self.fusion_encoder(image_features, text_features)  # N, retrieval_dim
        
        image_features = self.vision_encoder(entity_images)[1]  # received pooled output
        text_features = self.text_encoder(entity_token_ids, entity_attention_mask)[1]  # received pooled output
        # this is the last layer features before projection layer; without norm
        # N, D
        return self.fusion_encoder(
            torch.cat([image_features, text_features], dim=1)
        )  # N, retrieval_dim