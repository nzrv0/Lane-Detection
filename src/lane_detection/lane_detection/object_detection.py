from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import torch


def detector(image):
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    model.eval()
    with torch.no_grad():
        res = model(image)
        return res
