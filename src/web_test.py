from dataset import ObjectDataset
from model import FasterRcnn
from helpers import get_path
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import numpy as np

import streamlit as st


def get_data():
    image_path = "training"
    box_path = "data_object_label_2"
    dataset = ObjectDataset(image_path, box_path)
    data = dataset[2]
    return data


def load_model():
    model = FasterRcnn()
    model_path = get_path("models")
    model.load_state_dict(
        torch.load(
            model_path / "object_detection.pth",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()
    return model


def show_boxes(layer="roi"):
    data = get_data()

    image, gt_boxes, labels, gt_labels = (
        data["image"],
        data["cords"],
        data["labels"],
        data["gt_labels"],
    )

    image = image[None, :]

    gt_labels = torch.tensor(gt_labels)
    gt_boxes = gt_boxes.squeeze(0)

    model = load_model()
    rpn, roi = model(image, gt_boxes, gt_labels)

    max_el = torch.sort(roi["scores"] * 100, dim=0, descending=False)[1][:5]

    if layer != "roi":
        drawn_boxes = draw_bounding_boxes(
            image.squeeze().detach().cpu(),
            rpn["proposals"].detach().cpu(),
            colors="red",
            width=1,
        )
    else:
        drawn_boxes = draw_bounding_boxes(
            image.squeeze().detach().cpu(),
            roi["boxes"].detach().cpu(),
            colors="red",
            width=1,
        )
    img_with_boxes = F.to_pil_image(drawn_boxes)
    return img_with_boxes


def run_web_app():
    st.title("Detecting Objects With FasterRCNN")

    # img_rgb = show_boxes()
    from pathlib import Path
    import os

    path = Path("./training") / "image_2"
    # images = os.listdir("/home/nzrv/Dev/last_exam/src/training/image_2")
    
    # st.image(images[0], use_column_width=True)


run_web_app()
