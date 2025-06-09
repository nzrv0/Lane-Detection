from model import LaneNet
from dataset import LaneDataset
from helpers import get_device, show_image, get_path

import torch
import torchvision.transforms as T
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

device = get_device()

data = LaneDataset("TUSimple")

image, bin_lins = data[0]
image = image.permute(1, 2, 0).cpu().numpy()
bin_lins = bin_lins.permute(1, 2, 0).cpu().numpy()

image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(image.shape, bin_lins.shape)
img = cv.addWeighted(image, 0.3, bin_lins, 0.7, 0)
cv.imshow("image", img)

# Wait for a key
cv.waitKey(0)

# Distroy all the window open
cv.distroyAllWindows()

if __name__ == "__main__":
    # video()
    pass


def test_model():
    model = LaneNet().to(device)

    model.load_state_dict(
        torch.load(
            "./weights/first_model1.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    image, bin_lins, seg_lines = data[0]

    predict = model(image.unsqueeze(0))

    image = image.permute(1, 2, 0).cpu().numpy()
    bin_pred = predict[0].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255
    seg_pred = predict[1].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255

    seg_pred = np.hstack(seg_pred)

    seg_pred = bin_lins.cpu().squeeze(0).numpy()

    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(seg_pred, alpha=0.5)
    plt.show()
    # show_image(image)
    # show_image(bin_pred)
    # show_image(seg_pred, cmap="brg")
