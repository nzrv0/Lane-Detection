from torch.utils.data import Dataset
import os
from helpers import get_path, load_json, get_cords, get_device
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd

device = get_device()


class LaneDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = get_path(image_dir) / "VIL100"
        self.jpeg_path = "JPEGImages"
        self.segments = os.listdir(self.image_dir / self.jpeg_path)
        self.data = self.load_data()

        self.tfms = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                ),
            ]
        )
        self.tf = T.Compose(
            [
                T.Resize((512, 512)),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        data_item = self.data[index]

        dataset = self.load_data_spec(data_item)
        return dataset

    def load_data(self):
        data = []
        for segment in self.segments:
            dataset = [
                {
                    "image": f"{self.jpeg_path}/{segment}/{image}",
                    "line": f"Json/{segment}/{image}" + ".json",
                }
                for image in os.listdir(self.image_dir / self.jpeg_path / segment)
            ]
            data.extend(dataset)
        return data

    def load_data_spec(self, data):
        image = data["image"]
        line = data["line"]

        image_path = self.image_dir / image
        line_path = self.image_dir / line

        # get images
        image = Image.open(image_path)
        from helpers import show_image

        width, height = image.size
        tensor_image = self.tfms(image).to(device)
        import numpy as np

        image = tensor_image.permute(1, 2, 0).numpy()
        show_image(image)

        # load cords
        all_lines = load_json(line_path)
        points = [{"id": lane["id"], "lanes": lane["points"]} for lane in all_lines]
        cords = get_cords(points, width, height)
        cords = cords.squeeze()

        cords = Image.fromarray(cords)
        cords = self.tf(cords).to(device)
        # cords = []
        return tensor_image, cords

    def __len__(self):
        return len(self.segments)

        # def load_lines_images(self, data):
        #     image = data["image"]
        #     lane = data["line"]

        #     lines = pd.read_csv(
        #         self.image_dir / lane,
        #         header=None,
        #     )
        #     lines = lines.to_numpy()
        #     xy = []

        #     for item in lines:
        #         item = list(map(lambda x: float(x) if x else None, item[0].split(" ")))
        #         x = item[::2]
        #         y = item[1::2]
        #         x = list(filter(None, x))
        #         y = list(filter(None, y))
        #         xy.append([x, y])

        #     # get images
        #     image_path = self.image_dir / image
        #     image = Image.open(image_path)
        #     width, height = image.size
        #     tensor_image = self.tfms(image).to(device)

        #     # get lines

        #     cords = get_cords(xy, width, height)
        #     cords = cords.squeeze()
        #     cords = Image.fromarray(cords)
        #     cords = self.tf(cords).to(device)
        #     return tensor_image, cords

        # def load_lines_data(self, index):
        #     dataset = []
        #     lines_data = os.listdir(self.line)

        #     for _, item in enumerate(lines_data[::4]):
        #         line_path = self.line / item
        #         file_path, lines = load_json(line_path)
        #         file_path = "/".join(file_path.split("/")[1:])

        #         # read image
        #         image_path = self.image_dir / file_path
        #         image = Image.open(image_path)
        #         width, height = image.size
        #         tensor_image = self.tfms(image).to(device)

        #         # load lines
        #         cords = get_cords(lines, width, height)
        #         cords = Image.fromarray(cords)
        #         cords = self.tf(cords).to(device)
        #         dataset.append({"image": tensor_image, "cords": cords})
        #     return dataset
