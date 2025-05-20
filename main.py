from model import LaneNet
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from dataset import LaneDataset
from torch.optim import SGD
from helpers import get_path, get_device
from tqdm import tqdm


device = get_device()


def load_model(learning_rate=0.001):
    model = LaneNet(1).to(device)
    loss_fn = BCEWithLogitsLoss()
    optim = SGD(model.parameters(), learning_rate)
    return model, loss_fn, optim


def load_dataset():
    training = "data"
    # cords = "cords"
    traning_data = DataLoader(LaneDataset(training), batch_size=1, shuffle=True)
    return traning_data



def train_batch(x, y, model, loss, optim):
    model.train()
    pred = model(x)
    pred = torch.squeeze(pred).float()

    y = torch.squeeze(y).float()
    loss_fn = loss(y, pred)
    loss_fn.backward()
    optim.step()
    optim.zero_grad()
    return loss_fn.item()


if __name__ == "__main__":
    model_path = get_path("weights")
    model_name = model_path / "first_model.pth"
    traning_data = load_dataset()
    data = LaneDataset("data")

    model, loss_fn, optim = load_model()
    total_loss = []
    loss_per_data = []
    epochs = 5

    for epoch in range(epochs):
        for index, batch in enumerate(iter(traning_data)):
            image, line = batch
            loss = train_batch(image, line, model, loss_fn, optim)
            break
    #     loss_per_data.append(loss)
    # total_loss.append(sum(loss_per_data))

    # torch.save(model.state_dict(), model_name)
