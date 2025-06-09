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
        self.binary_segmentation = UnetDecoder(2)
        self.pix_embeding = UnetDecoder(3)

        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        down1, down2, down3, down4, down5 = self.encoder(x)

        binary_segmentation = self.binary_segmentation(
            down1, down2, down3, down4, down5
        )

        pix_embeding = self.pix_embeding(down1, down2, down3, down4, down5)

        binary_segmentation = torch.argmax(
            F.softmax(binary_segmentation, dim=1), dim=1, keepdim=True
        ).float()

        pix_embeding = self.sigmoid(pix_embeding)

        return binary_segmentation, pix_embeding
