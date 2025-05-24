from model import LaneNet
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from dataset import LaneDataset
from torch.optim import Adam
from helpers import get_path, get_device
from tqdm import tqdm
from loss import DiscriminativeLoss, FocalLoss

device = get_device()


def load_dataset():
    training = "TUSimple"
    traning_data = DataLoader(LaneDataset(training), batch_size=3)
    return traning_data


def load_model(learning_rate=5e-4):
    model = LaneNet().to(device)
    optim = Adam(model.parameters(), learning_rate)
    return model, optim


def calc_loss(pred_segmentats, binary_segments, pred_embeding, instances):
    k_binary = 10
    k_instance = 0.3
    k_dist = 1.0

    segmentation_loss = FocalLoss()
    binary_loss = segmentation_loss(pred_segmentats, binary_segments)

    embedding_loss = DiscriminativeLoss()
    var_loss, dist_loss = embedding_loss(pred_embeding, instances)

    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    print(binary_loss, instance_loss)
    return total_loss, binary_loss, instance_loss


def train_batch(image, bin_lines, seg_lines, model, optim):
    model.train()
    segmentation, pix_embeding = model(image)

    optim.zero_grad()

    total_loss = calc_loss(segmentation, bin_lines, pix_embeding, seg_lines)
    total_loss[0].backward()
    optim.step()

    # 0 segmentaiton loss, 1 binary loss
    return total_loss[0].item(), total_loss[1].item()


if __name__ == "__main__":
    model_path = get_path("weights")
    model_name = model_path / "first_model.pth"
    traning_data = load_dataset()
    data = LaneDataset("TUSimple")

    model, optim = load_model()
    total_loss = []
    loss_per_data = []
    epochs = 5

    for epoch in range(epochs):
        for index, batch in enumerate(tqdm(iter(traning_data))):
            image, bin_lines, seg_lines = batch
            loss = train_batch(image, bin_lines, seg_lines, model, optim)
            break
        loss_per_data.append(loss)
    total_loss.append(sum(loss_per_data))

    # torch.save(model.state_dict(), model_name)
