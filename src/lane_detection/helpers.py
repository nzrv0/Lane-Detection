from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import torch


def get_path(subpath: str) -> Path:
    path = Path(".")
    path = path / subpath
    return path


def load_json(path):
    data = ""
    with open(path, "r") as fs:
        data += fs.read()
    data = json.loads(data)
    return data["annotations"]["lane"]


def show_image(image, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.show()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_cords(lines, w, h, mode="binary"):
    background = np.zeros((h, w))
    lanes = []
    for x in lines["x"]:
        lanes.append(list(zip(x, lines["y"])))
    point = []
    color = 255
    for index, item in enumerate(lanes[::-1]):
        for xy in item:
            if xy[0] != -2:
                x = round(abs(int(xy[0]))) - 1
                y = round(abs(int(xy[1]))) - 1
                if len(point) == 0:
                    cv.line(background, [x, y], [x, y], (color), 10)
                else:
                    cv.line(background, point[-1], [x, y], (color), 10)
                point.append([x, y])
        if mode != "binary":
            color -= 30
        point = []
    return background
