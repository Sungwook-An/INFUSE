import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
    def __init__(self, image_dim, text_dim, reduction_ratio=16):
        super(ChannelWiseAttention, self).__init__()
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, image_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(image_dim, image_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(image_dim // reduction_ratio, image_dim),
            nn.Sigmoid()
        )
        
    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat (Tensor): Shape (B, C, H, W)
            text_feat (Tensor): Shape (B, D)
            
        Returns:
            Tensor of shape (B, C, H, W): Attention-modulated image features
        """
        B, C, H, W = image_feat.shape
        
        image_feat_proj = self.image_proj(image_feat)  # (B, C, H, W)
        pooled_feat = F.adaptive_avg_pool2d(image_feat_proj, 1).view(B, C)
        
        text_proj = self.text_proj(text_feat)  # (B, C)
        fusion = pooled_feat + text_proj # (B, C)
        
        weights = self.fc(fusion).view(B, C, 1, 1)  # (B, C, 1, 1)
        out = image_feat * weights
        
        return out
    

class CrossAttention(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=512):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(text_dim, hidden_dim)
        self.key_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, image_dim, kernel_size=1)

    def forward(self, image_feat, text_feat):
        """
        Args:
            image_feat: Tensor of shape (B, C, H, W)
            text_feat: Tensor of shape (B, D)
        Returns:
            attended_image_feat: Tensor of shape (B, C, H, W)
        """
        B, C, H, W = image_feat.shape

        Q = self.query_proj(text_feat).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        K = self.key_proj(image_feat)  # (B, H, H, W)
        V = self.value_proj(image_feat)  # (B, H, H, W)

        # Compute attention scores
        attn_scores = (Q * K).sum(dim=1, keepdim=True) / (K.shape[1] ** 0.5)  # (B, 1, H, W)
        attn_weights = F.softmax(attn_scores.view(B, -1), dim=-1).view(B, 1, H, W)  # (B, 1, H, W)

        # Apply attention
        weighted_value = attn_weights * V  # (B, H, H, W)
        attended = self.output_proj(weighted_value)  # (B, C, H, W)
        return attended


########################################################
#############  Checking the implementation #############
########################################################
if __name__ == "__main__":
    batch_size = 4
    image_dim = 640
    height = 5
    width = 5
    text_dim = 640

    image_feat = torch.randn(batch_size, image_dim, height, width)
    text_feat = torch.randn(batch_size, text_dim)

    print("== Channel-wise Cross Attention ==")
    channel_attn = ChannelWiseAttention(image_dim=image_dim, text_dim=text_dim)
    out1 = channel_attn(image_feat, text_feat)
    print("Output shape (Channel-wise):", out1.shape)

    print("\n== General Cross Attention ==")
    cross_attn = CrossAttention(image_dim=image_dim, text_dim=text_dim)
    out2 = cross_attn(image_feat, text_feat)
    print("Output shape (Cross Attention):", out2.shape)
########################################################
############  End of implementation check ##############
########################################################