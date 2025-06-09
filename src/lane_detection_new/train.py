from model import Unet
from dataset import LaneDataset
from helpers import get_path, get_device

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam

from tqdm import tqdm

device = get_device()


def load_dataset():
    training = "TUSimple"
    # traning_data = DataLoader(LaneDataset(training), batch_size=1, shuffle=True)
    traning_data = DataLoader(LaneDataset(training), batch_size=1)

    return traning_data


def load_model(learning_rate=5e-4):
    model = Unet().to(device)
    optim = Adam(model.parameters(), learning_rate)
    loss = BCEWithLogitsLoss(pos_weight=torch.tensor([0.75], device=device))
    # loss = CrossEntropyLoss(weight=torch.tensor([0.75], device=device))

    return model, optim, loss


def train_batch(image, bin_lines, model, optim, loss_fn):
    model.train()

    pred = model(image)
    # from helpers import show_image
    # show_image(pred.squeeze().detach().numpy())
    loss_fn = loss_fn(bin_lines, pred)

    loss_fn.backward()

    optim.step()
    optim.zero_grad()

    return loss_fn.item()


if __name__ == "__main__":
    model_path = get_path("weights")

    traning_data = load_dataset()

    model, optim, loss_fn = load_model()
    total_loss = []
    loss_per_data = []
    loss_per_epoch = []

    epochs = 10

    for epoch in range(epochs):
        for index, batch in enumerate(tqdm(iter(traning_data))):
            image, bin_lines = batch
            loss = train_batch(image, bin_lines, model, optim, loss_fn)

            loss_per_data.append(loss)
            print(loss)
        loss_per_epoch.append(loss_per_data)

        model_name = model_path / f"model{epoch}.pth"
        torch.save(model.state_dict(), model_name)

    total_loss.append(sum(loss_per_data))
