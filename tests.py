from PIL import Image
import matplotlib.pyplot as plt
from helpers import get_path
import cv2 as cv
from memory_profiler import profile
from model import LaneNet
import torch
from dataset import LaneDataset
from helpers import show_image, get_device
import numpy as np

device = get_device()
import torchvision.transforms.functional as TF


def evaluate():
    data = LaneDataset("data")
    model = LaneNet(1).to(device)
    model.load_state_dict(
        torch.load(
            "/home/nzrv/Dev/cv/lane_detection/weights/first_model.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    image, lines = data[0]
    ground_truth = Image.open(
        "/home/nzrv/Dev/cv/lane_detection/data/VIL100/JPEGImages/ddd/00006.jpg"
    )
    # image = TF.to_tensor(ground_truth)
    predict = model(image.unsqueeze(0))
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    # image = (
    #     predict.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8())
    # )

    # test_video = "/home/nzrv/Dev/cv/lane_detection/test/3602494-hd_1920_1080_30fps.mp4"
    # cap = cv.VideoCapture(test_video)
    # Create a window that you can resize
    # cv.namedWindow("Video", cv.WINDOW_NORMAL)

    # Set a specific size
    # cv.resizeWindow("Video", 1200, 800)  # width x height

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     # edges = cv.Canny(frame, 100, 200)

    #     frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    #     predict = model(image.unsqueeze(0))
    #     print(predict)
    #     frame = (
    #         predict.squeeze(0)
    #         .permute(1, 2, 0)
    #         .detach()
    #         .cpu()
    #         .numpy()
    #         .astype(np.uint8())
    #     )
    #     # gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #     cv.imshow("Video", frame)
    #     if cv.waitKey(1) == ord("q"):
    #         break

    # cap.release()
    # cv.destroyAllWindows()

    # lines = lines.permute(1, 2, 0).detach().cpu().numpy()
    # image = predict.permute(1, 2, 0).detach().cpu().numpy()
    # image = Image.fromarray(lines).convert("L")

    # image = image[1].unsqueeze(0)
    # predict = model(image)

    # predict = torch.argmax(predict, dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
    # predict = torch.sigmoid(predict)
    # predict = torch.squeeze(predict).detach().numpy()
    # threshold = 0.5
    # pred_mask = (predict).float().detach().squeeze().cpu().numpy()  # Shape: (H, W)

    # image = cv.imread("/home/nzrv/Dev/cv/lane_detection/data/data.mp4/00180.jpg")

    # cv.circle(image, (447, 63), 5, (0, 0, 255), -1)
    # cv.imshow("windows", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # ground_truth = np.array(ground_truth)
    # ground_truth = np.resize(ground_truth, (512, 512))
    # ground_truth = (ground_truth > 0).astype(np.uint8)
    # and_mask = np.logical_and(pred_mask, ground_truth).astype(np.uint8)
    show_image(image)


import sys

if __name__ == "__main__":
    evaluate()
    # np.set_printoptions(threshold=sys.maxsize)
    # data = LaneDataset("training", "cords")
    # image, lines = data[0][0].items()
    # line = lines[1].permute(1, 2, 0).numpy()
    # # print(np.where(lines[1] == 1))
    # image = image[1].permute(1, 2, 0).numpy()

    # plt.imshow(line, cmap="gray")
    # # plt.imshow(image, cmap="gray")
    # plt.show()
