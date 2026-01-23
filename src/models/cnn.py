# CNN Feature Extractor for Surgical Video Analysis
# Uses ResNet-50 pretrained on ImageNet as backbone
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNet50_FeatureExtractor(nn.Module):
    """
    ResNet-50 feature extractor for surgical video frames.

    This CNN extracts visual features from each video frame. These features
    capture what's happening in the image (tools, anatomy, actions).

    Architecture:
    - Uses ResNet-50 pretrained on ImageNet (1M natural images)
    - Removes final classification layer (we don't need 1000 ImageNet classes)
    - Outputs 2048-dimensional feature vector per frame

    Why pretrained?
    - ImageNet pretraining teaches general visual features (edges, textures, shapes)
    - These transfer well to medical imaging with fine-tuning
    - Training from scratch would require much more surgical data

    Freezing strategy:
    - freeze=True: All CNN weights frozen, only LSTM learns
      → Fast training, uses ImageNet features as-is
    - freeze=False: Layer4 unfrozen, earlier layers frozen
      → Layer4 adapts to surgical domain while preserving low-level features
    """

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        # Load pretrained ResNet-50
        # Using new weights API (pretrained= is deprecated)
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2  # Best available weights
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        # Remove the final classification layer (fc)
        # Keep: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool
        # Remove: fc (1000-class classifier)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Output dimension after global average pooling
        self.feature_dim = 2048

        # Freezing strategy
        if freeze:
            # Freeze ALL parameters - CNN is pure feature extractor
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Freeze early layers, unfreeze layer4
            # Early layers: generic features (edges, textures) - keep frozen
            # Layer4: high-level features - allow adaptation to surgical domain
            # Note: In nn.Sequential, layer4 is at index 7 (names are "7.xxx")
            for name, param in self.backbone.named_parameters():
                if name.startswith("7."):  # layer4 is index 7 in Sequential
                    param.requires_grad = True  # Allow surgical domain adaptation
                else:
                    param.requires_grad = False  # Keep generic features frozen

    def forward(self, x):
        """
        Extract features from a sequence of video frames.

        Args:
            x: [B, T, C, H, W] batch of video sequences
               B = batch size
               T = sequence length (number of frames)
               C = channels (3 for RGB)
               H, W = height, width (should be 224x224 for ResNet)

        Returns:
            features: [B, T, 2048] feature vectors for each frame
        """
        B, T, C, H, W = x.shape

        # Merge batch and time dimensions for efficient CNN processing
        # CNN processes each frame independently
        x = x.view(B * T, C, H, W)

        # Extract features through ResNet backbone
        feats = self.backbone(x)  # [B*T, 2048, 1, 1]

        # Reshape back to [B, T, feature_dim]
        feats = feats.view(B, T, self.feature_dim)

        return feats
