from model import LaneNet
from dataset import LaneDataset
from helpers import get_device, show_image, get_path

import torch
import torchvision.transforms as T
import numpy as np
import cv2 as cv
from PIL import Image

device = get_device()

data = LaneDataset("TUSimple")
model = LaneNet().to(device)
model.load_state_dict(
    torch.load(
        "./weights/first_model.pth",
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

# show_image(image)
# show_image(bin_pred)
# show_image(seg_pred, cmap="brg")
tfms = T.Compose(
    [
        T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ]
)


def video():
    path = get_path("videos")
    video_path = path / "test.mp4"
    cap = cv.VideoCapture(video_path)
    cv.namedWindow("Video", cv.WINDOW_NORMAL)

    cv.resizeWindow("Video", 1200, 800)

    while cap.isOpened():
        ret, frame = cap.read()
        # expand_dims
        image = Image.fromarray(frame)
        frame = tfms(image).to(device)
        predict = model(frame.unsqueeze(0))
        seg_pred = predict[1].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255
        bin_pred = predict[0].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255

        seg_pred = np.hstack(seg_pred)

        cv.imshow("Video", bin_pred)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    video()
