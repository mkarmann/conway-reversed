import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import os

from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

LR = 1e-4
BATCH_SIZE = 128
STEPS_PER_EPOCH = 128
EPOCHS = 32
GOL_DELTA = 1
RUN_NAME = time.strftime("%Y_%m_%d_%H_%M_%S") + '_GoL_delta_' + str(GOL_DELTA)
SNAPSHOTS_DIR = '../out/training/snapshots/{}'.format(RUN_NAME)
TENSORBOARD_LOGS_DIR = '../out/training/logs'


def step(state: np.array):
    neighbour_sum = \
        np.roll(state, -1, axis=0) + \
        np.roll(state, 1, axis=0) + \
        np.roll(state, -1, axis=1) + \
        np.roll(state, 1, axis=1) + \
        np.roll(np.roll(state, -1, axis=0), -1, axis=1) + \
        np.roll(np.roll(state, -1, axis=0), 1, axis=1) + \
        np.roll(np.roll(state, 1, axis=0), -1, axis=1) + \
        np.roll(np.roll(state, 1, axis=0), 1, axis=1)
    out = np.zeros(state.shape, dtype=np.int)
    out[neighbour_sum == 3] = 1
    out[np.logical_and(neighbour_sum == 2, state == 1)] = 1
    return out


def plot(state: np.array):
    plt.imshow(state.astype(np.float))
    plt.show()


def create_random_board(shape=(25, 25), warmup_steps=5):
    factor = np.random.uniform(0.01, 0.99, (1, ))
    state = (np.random.uniform(0.0, 1.0, shape) > factor).astype(np.int)

    for i in range(warmup_steps):
        state = step(state)

    return state


def create_training_sample(shape=(25, 25), warmup_steps=5, delta=1):

    while True:
        start = create_random_board(shape, warmup_steps)
        end = start
        for i in range(delta):
            end = step(end)

        if np.any(end):
            return {
                "start": start,
                "end": end,
                "delta": delta
            }


class ResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.main = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, 3, padding=0, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, 3, padding=0,  bias=False)
        )

    def forward(self, x):
        return self.main(x) + x[:, :, 2:-2, 2:-2]


class GoLDataset(Dataset):
    def __init__(self, shape=(25, 25), warmup_steps=5, delta=1, size=1024, outline_size=0):
        self.shape = shape
        self.warmup_steps = warmup_steps
        self.delta = delta
        self.size = size
        self.outline_size = outline_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = create_training_sample(self.shape, self.warmup_steps, self.delta)
        start = sample["start"]
        end = sample["end"]
        predicted_mask = (np.random.uniform(0.0, 1.0, self.shape) > np.random.uniform(0.0, 1.0, (1, ))).astype(np.int)

        # there needs to be a cell left to predict
        if np.sum(predicted_mask) == self.shape[0] * self.shape[1]:
            predicted_mask[np.random.randint(0, self.shape[0], (1, )), np.random.randint(0, self.shape[1], (1, ))] = 0

        input_dead = (1 - end).astype(np.float)
        input_alive = end.astype(np.float)
        input_unpredicted = (1 - predicted_mask).astype(np.float)
        input_predicted_dead = ((1 - start) * predicted_mask).astype(np.float)
        input_predicted_alive = (start * predicted_mask).astype(np.float)
        sample_input = np.stack([
            input_dead,
            input_alive,
            input_unpredicted,
            input_predicted_dead,
            input_predicted_alive],
            axis=2
        )

        sample_target = start

        # outlining
        if self.outline_size > 0:
            tiles_y = ((self.outline_size - 1) // self.shape[0]) * 2 + 2 + 1
            tiles_x = ((self.outline_size - 1) // self.shape[1]) * 2 + 2 + 1
            offset_y = self.shape[0] - ((self.outline_size - 1) % self.shape[0]) - 1
            offset_x = self.shape[1] - ((self.outline_size - 1) % self.shape[1]) - 1

            sample_input = np.tile(sample_input, (tiles_y, tiles_x, 1))
            sample_input = sample_input[
                           offset_y:(offset_y + self.shape[0] + 2 * self.outline_size),
                           offset_x:(offset_x + self.shape[1] + 2 * self.outline_size)
                           ]

        return {
            "input": sample_input.transpose((2, 0, 1)),
            "mask": np.expand_dims(predicted_mask.astype(np.float), 3).transpose((2, 0, 1)),
            "target": sample_target
        }


class GoLModule(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(5, channels, 1, bias=False),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 2, 1)
        )

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":
    device = torch.device('cuda')

    dataset = GoLDataset(delta=GOL_DELTA, size=STEPS_PER_EPOCH * BATCH_SIZE, outline_size=4*2)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=1,
                        pin_memory=True)

    net = GoLModule()
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    cel = nn.CrossEntropyLoss()

    if not os.path.isdir(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR)
    # train_writer = SummaryWriter(log_dir=TENSORBOARD_LOGS_DIR + '/' + RUN_NAME, comment=RUN_NAME)

    for epoch in range(EPOCHS):
        batch_iter = iter(loader)
        for step in range(STEPS_PER_EPOCH):
            batch = next(batch_iter)

            net.zero_grad()
            torch_input = batch['input'].float().to(device)
            torch_target = batch['target'].long().to(device)
            prediction = net(torch_input)
            loss = cel(prediction, torch_target)

            loss.backward()
            optimizer.step()

            print('Epoch {} - Step {} - loss {:.5f}'.format(epoch + 1, step + 1, loss.item()))

    print("Done!")
