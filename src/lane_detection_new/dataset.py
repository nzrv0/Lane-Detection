from torch.utils.data import Dataset
from helpers import get_path, get_device
import torchvision.transforms as T
from PIL import Image
import json

device = get_device()


class LaneDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = get_path(image_dir) / "train_set"
        self.dataset = self.get_data()

        self.tfms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
                T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], inplace=True),
            ]
        )
        self.tfms2 = T.Compose(
            [
                T.ToTensor(),
                T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.1),
            ]
        )

    def __getitem__(self, index):
        dataset = self.dataset[index]
        dataset = self.load_data_spec(dataset)
        return dataset

    def get_data(self):
        # cords = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]
        cords = ["label_data_0313.json"]

        data = []

        for cord in cords:
            with open(self.image_dir / cord, "r") as fs:
                dd = [item[:-1] for item in fs.readlines()]
                parse = json.dumps(dd)
                parse = json.loads(parse)
                for item in parse:
                    data_item = json.loads(item)
                    if data_item["raw_file"].split("/")[-1] == "20.jpg":
                        data.append(
                            {
                                "image": self.image_dir / data_item["raw_file"],
                                "segments": self.image_dir
                                / (
                                    "seg_label/"
                                    + "/".join(data_item["raw_file"].split("/")[1:-1])
                                    + "/20.png"
                                ),
                            }
                        )
        return data

    def load_data_spec(self, data):
        image = data["image"]
        segments = data["segments"]

        # get images
        image = Image.open(image)
        tensor_image = self.tfms(image).to(device)

        # load cords
        segments = Image.open(segments).convert("L")
        tensor_segments = self.tfms2(segments).to(device)

        return tensor_image, tensor_segments

    def __len__(self):
        return len(self.dataset)
