from torch.utils.data import Dataset
from helpers import get_path, get_cords, get_device
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
                T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )
        self.tf = T.Compose(
            [
                T.Resize((512, 256), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        dataset = self.dataset[index]
        dataset = self.load_data_spec(dataset)
        return dataset

    def get_data(self):
        # LOAD THIS WHEN TRAINING
        # cords = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]

        cords = ["label_data_0313.json"]
        data = []
        for cord in cords:
            with open(self.image_dir / cord, "r") as fs:
                dd = [item[:-1] for item in fs.readlines()]
                parse = json.dumps(dd)
                parse = json.loads(parse)
                for item in parse:
                    item = json.loads(item)
                    data.append(
                        {
                            "image": self.image_dir / item["raw_file"],
                            "lanes": {"x": item["lanes"], "y": item["h_samples"]},
                        }
                    )
        return data

    def load_data_spec(self, data):
        image = data["image"]
        lanes = data["lanes"]

        # get images
        image = Image.open(image)
        width, height = image.size
        tensor_image = self.tfms(image).to(device)

        # load cords
        seg_cords = get_cords(lanes, width, height, mode="segmentation")
        seg_cords = seg_cords.squeeze()
        seg_cords = Image.fromarray(seg_cords)
        seg_cords = self.tf(seg_cords).to(device)

        bin_seg = get_cords(lanes, width, height)
        bin_seg = bin_seg.squeeze()
        bin_seg = Image.fromarray(bin_seg)
        bin_seg = self.tf(bin_seg).to(device)

        return tensor_image, bin_seg, seg_cords

    def __len__(self):
        return len(self.dataset)


