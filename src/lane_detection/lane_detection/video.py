from helpers import get_path, get_device

import torchvision.transforms as T

import torch

import numpy as np
import cv2 as cv
from PIL import Image

device = get_device()


def load_data():
    pass


def video():
    # model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    tfms = T.Compose(
        [
            T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )
    path = get_path("videos")
    video_path = path / "test.mp4"
    cap = cv.VideoCapture(video_path)
    cv.namedWindow("Video", cv.WINDOW_NORMAL)

    cv.resizeWindow("Video", 1200, 800)

    while cap.isOpened():
        ret, frame = cap.read()
        # expand_dims
        from object_detection import detector

        # image = Image.fromarray(frame)
        im = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
        resoult = detector(im)
        for item in resoult[0]["boxes"]:
            res = item
            xl, yl = res[0].item(), res[1].item()
            xr, yr = res[2].item(), res[3].item()
            xl, yl = int(xl), int(yl)
            xr, yr = int(xr), int(yr)
            # cv.addText(
            #     frame,
            #     res["name"],
            # )
            cv.rectangle(frame, (xl, yl), (xr, yr), (255, 255, 255), 3)
        # frame = tfms(image).to(device)
        # predict = model(frame.unsqueeze(0))
        # seg_pred = predict[1].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255
        # bin_pred = predict[0].detach().cpu().squeeze(0).permute(2, 1, 0).numpy() * 255

        # seg_pred = np.hstack(seg_pred)

        cv.imshow("Video", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


video()
