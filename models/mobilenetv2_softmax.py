import torch.nn as nn
import torch.nn.functional as F
from  torchvision import models
import torchvision.transforms as transforms

class MobileNetV2Softmax(nn.Module):
    def __init__(self, num_class, pretrain = False):
        super(MobileNetV2Softmax, self).__init__()
        self.model =  models.mobilenet_v2(pretrained = pretrain)
        self.model.classifier = nn.Linear(1280, num_class)
        self.softmax = nn.Softmax()

    def forward(self, image):
        fc = self.model(image)
        output = self.softmax(fc)

        return output