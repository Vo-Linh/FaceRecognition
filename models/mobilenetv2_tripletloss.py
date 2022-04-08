import torch
import torch.nn as nn
from  torchvision import models
import torch.nn.functional as F

class MobileNetV2Triplet(nn.Module):
    def __init__(self, embedding_dim = 512, pretrain = False):
        super(MobileNetV2Triplet, self).__init__()
        self.model =  models.mobilenet_v2(pretrained = pretrain)

        # Output embedding
        self.model.classifier = nn.Linear(1280, embedding_dim, bias=False)

    def forward(self, image):
        embedding = self.model(image)
        embedding = F.normalize(embedding, p = 2, dim = 1)

        return embedding