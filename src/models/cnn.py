# we are gonna seperate the models and there use
# CNN and LSTM:seeing and reasoning over time
import torch
import torch.nn as nn
import torchvision.models as models

# our CNN model will play the role of a feature extractor at the first stage
# captures data and patterns visually in the phrames
class ResNet50_FeatureExtractor(nn.Module):
    """
    ResNet-50 feature extractor.
    
    Takes a sequence of images and outputs a sequence of feature vectors.
    """

    def __init__(self , pretrained = True, freeze = True):
        super().__init__()

        # load the pretrained resnet model
        resnet = models.resnet50(pretrained = pretrained)

        # remove the final classification layer (fc)
        # keep everything up to average pooling 
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = 2048

        # freeze backbone 
        if not freeze:
            for name, param in self.backbone.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True   # allow adaptation
                else:
                    param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        # need to alter the dimensions
        B, T,C, H, W= x.shape
        # Merge batch and time dimensions
        x = x.view(B * T, C, H, W)
        # Pass through ResNet backbone
        feats = self.backbone(x)
        # Remove spatial dimensions
        feats = feats.view(B, T, self.feature_dim)

        return feats  

        
