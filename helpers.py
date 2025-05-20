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


def bird_eye(image):
    rows, cols, chn = image.shape
    gap = 60
    pts1 = np.float32(
        [
            [cols / 2 - gap, 0],
            [cols / 2 + gap, 0],
            [0, rows],
            [cols, rows],
        ]
    )

    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    output = cv.warpPerspective(
        image,
        M,
        (cols, rows),
        flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR | cv.WARP_FILL_OUTLIERS,
    )

    return output


def max_array_show():
    import sys

    np.set_printoptions(threshold=sys.maxsize)


def set_ones_around_point(array, x, y, radius=25):
    rows, cols = array.shape
    for i in range(max(0, x - radius), min(rows, x + radius + 1)):
        for j in range(max(0, y - radius), min(cols, y + radius + 1)):
            if i == x and j == y:
                continue
            array[x:i, y:j] = 1

    return array


def get_cords(lines, w, h):
    background = np.zeros((h, w))
    points = []
    mean = w / 2
    gap = mean / 2
    for line in lines:
        point = []
        for index, item in enumerate(line["lanes"]):
            x = round(abs(item[0])) - 1
            y = round(abs(item[1])) - 1
            point.append([x, y])
            background[y, x] = 1
            set_ones_around_point(background, y, x)

        # point = sorted(point, key=lambda x: x[0])

        # if mean - gap < point[-1][0] < mean + gap:
        # points.append([point[0], point[-1]])
        # for item in point:

    # points = sorted(points, key=lambda x: x[0])
    # idd = 0
    # while len(points) - 1 > idd:
    #     pp = np.array(points[idd])

    #     pd = points[idd + 1]
    #     pp[0][0] -= 100
    #     pd[1][0] += 200
    #     t1 = pp[0]
    #     b1 = pp[1]
    #     t2 = pd[0]
    #     b2 = pd[1]
    #     pr = np.array(
    #         [
    #             b1,
    #             b2,
    #             t1,
    #             t2,
    #         ],
    #         np.int32,
    #     )
    #     idd += 1
    #     cv.fillPoly(background, [pr], color=(255, 255, 255))
    #     background = background.astype(np.uint8)
    return background


def load_json(path):
    data = ""
    with open(path, "r") as fs:
        data += fs.read()
    data = json.loads(data)
    return data["annotations"]["lane"]


def show_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
