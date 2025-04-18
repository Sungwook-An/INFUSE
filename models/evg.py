import torch
import torch.nn as nn
import torch.nn.functional as F


class EVGNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, top_k=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, class_embedding, entity_embeddings):
        """
        Args:
            class_embedding (Tensor): Shape (B, D), output from text encoder (E_t)
            entity_embeddings (Tensor): Shape (B, N, D), entity features from JSON (20 per class)
            
        Returns:
            Tensor of shape (B, D): final entity representation (v_e)
        """
        print(class_embedding.shape)
        print(entity_embeddings.shape)
        # B, N, D = entity_embeddings.shape
        
        # Project Q, K, V
        Q = self.query_proj(class_embedding).unsqueeze(1) # (B, 1, H)
        K = self.key_proj(entity_embeddings)              # (B, N, H)
        V = self.value_proj(entity_embeddings)             # (B, N, H)
        
        # Compute attention
        # attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # (B, 1, N)
        attn_logits = torch.einsum('bhd,bnd->bhn', Q, K) / (self.hidden_dim ** 0.5) # (B, 1, N)
        attn_scores = F.softmax(attn_logits, dim = -1) # (B, 1, N)
        
        # Top-k selection
        topk_scores, topk_indices = torch.topk(attn_scores, self.top_k, dim=-1) # (B, 1, top_k)
        
        topk_indices = topk_indices.squeeze(1) # (B, top_k)
        topk_scores = topk_scores.squeeze(1)   # (B, top_k)
        
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, self.top_k, self.hidden_dim)
        
        # Gather top-k value vectors and and apply weights
        topk_values = torch.gather(V, 1, expanded_indices) # (B, top_k, H)
        weighted_sum = torch.sum(topk_values * topk_scores.unsqueeze(-1), dim=1)
        
        # Final projection
        final_vector = self.output_proj(weighted_sum)
        
        return final_vector
    

########################################################
#############  Checking the implementation #############
########################################################
if __name__ == "__main__":
    from models.text_encoder import TextEncoder
    import json
    
    class_name = "apple"
    
    entities = [
        "Glossy skin with light reflections",
        "Bright red coloration on the majority of apples",
        "Yellow-green hue on some apple surfaces",
        "Presence of a short brown stem",
        "Smooth, unblemished surface",
        "Slightly asymmetrical contours in some apples",
        "Round to oval body shape",
        "Faint vertical streaks or gradients in color",
        "Dark red apples with deep shine",
        "Visible stem cavity at the top",
        "Blush-like pink highlights on red apples",
        "Flat or dimpled bottom in some examples",
        "Color fade toward the base (lighter near bottom)",
        "Pale speckles on skin of lighter apples",
        "Green apples with a warmer yellow tone",
        "Some apples have a leaf attached to the stem",
        "Pairs or clusters of apples (not always solitary)",
        "Highly saturated red tones in some examples",
        "Visible surface shading indicating 3D roundness",
        "Distinct color contrast between highlight and shadow areas"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = TextEncoder(encoder_name='bert', projection_dim=512, device=device)
    
    class_emb = encoder([class_name]).to(device)
    entity_embs = encoder(entities).unsqueeze(0).to(device)
    
    evg = EVGNetwork(input_dim=512, top_k=5, hidden_dim=512, output_dim=640).to(device)
    
    ve = evg(class_emb, entity_embs)
    print("Entity vector shape:", ve.shape)
########################################################
############  End of implementation check ##############
########################################################