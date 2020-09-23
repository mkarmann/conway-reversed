import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader

LR = 1e-4
BATCH_SIZE = 128
STEPS_PER_EPOCH = 128
EPOCHS = 32
GOL_DELTA = 1
RUN_NAME = time.strftime("%Y_%m_%d_%H_%M_%S") + '_GoL_delta_' + str(GOL_DELTA)
SNAPSHOTS_DIR = '../out/training/snapshots/{}'.format(RUN_NAME)


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


class GoLDataset(Dataset):
    def __init__(self, shape=(25, 25), warmup_steps=5, delta=1, size=1024):
        self.shape = shape
        self.warmup_steps = warmup_steps
        self.delta = delta
        self.size = size

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

        target_dead = (1 - start).astype(np.float)
        target_alive = start.astype(np.float)
        sample_target = np.stack([
            target_dead,
            target_alive],
            axis=2
        )

        return {
            "input": sample_input,
            "mask": predicted_mask.astype(float),
            "target": sample_target
        }


class GoLModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(5, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":

    dataset = GoLDataset(delta=GOL_DELTA, size=STEPS_PER_EPOCH * BATCH_SIZE)
    print(dataset[0])
    exit()
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=1,
                        pin_memory=True)

    if not os.path.isdir(SNAPSHOTS_DIR):
        os.makedirs(SNAPSHOTS_DIR)

    for epoch in range(EPOCHS):
        batch_iter = iter(loader)
        for step in range(STEPS_PER_EPOCH):
            pass