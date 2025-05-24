import torch
from torch import nn
from unet import UnetEncoder, UnetDecoder
from helpers import get_device
import torch.nn.functional as F

device = get_device()


class LaneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UnetEncoder(3)
        self.segmentation = UnetDecoder(1)
        self.pix_embeding = UnetDecoder(5)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        down1, down2, down3, down4, down5 = self.encoder(x)

        segmentation = self.segmentation(down1, down2, down3, down4, down5)
        segmentation = torch.argmax(F.softmax(segmentation, dim=1), dim=1, keepdim=True)
        segmentation = segmentation.float()
        pix_embeding = self.pix_embeding(down1, down2, down3, down4, down5)

        pix_embeding = self.sigmoid(pix_embeding)

        return segmentation, pix_embeding
