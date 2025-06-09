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
