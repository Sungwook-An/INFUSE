import torch
import torch.nn as nn

from models.resnet12 import Res12
from models.text_encoder import TextEncoder
from models.evg import EVG
from models.attention_modules import EntityGuidedCrossAttention
from models.classifier import MatchingNetworkClassifier

from modules.prompt_utils import generate_entity_tokens

class INFUSEModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 모듈 구성
        self.image_encoder = Res12(**args.image_encoder_config)
        self.text_encoder = TextEncoder(**args.text_encoder_config)
        self.evg = EVG(args)
        self.cross_attn = EntityGuidedCrossAttention(args)
        self.classifier = MatchingNetworkClassifier(args)
        
        self.args = args

    def forward(self, support_images, support_labels, query_images, class_names):
        """
        Args:
            support_images: (N*K, C, H, W)
            support_labels: (N*K)
            query_images:   (N*Q, C, H, W)
            class_names:    List[str] of length N

        Returns:
            logits: (N*Q, N)
        """
        # 1. 이미지 인코딩
        support_feat = self.image_encoder(support_images)  # [N*K, D]
        query_feat = self.image_encoder(query_images)      # [N*Q, D]

        # 2. 텍스트 인코딩: class_name -> entity tokens -> token embeddings
        entity_vectors = []
        for cls_name in class_names:
            entity_tokens = generate_entity_tokens(cls_name)  # ex: ["fur", "wild", ...]
            token_embeddings = self.text_encoder(entity_tokens)  # [L, D]
            class_embedding = self.text_encoder([cls_name])[0]   # [D]

            entity_vector = self.evg(token_embeddings, class_embedding)  # [D]
            entity_vectors.append(entity_vector)

        entity_vectors = torch.stack(entity_vectors, dim=0)  # [N, D]

        # 3. Cross-attention with entity vector
        support_feat = self.cross_attn(support_feat, entity_vectors, support_labels)

        # 4. Matching network classifier
        logits = self.classifier(query_feat, support_feat, support_labels)

        return logits
