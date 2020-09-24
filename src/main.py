import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
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
STEPS_PER_EPOCH = 1024
EPOCHS = 64
GOL_DELTA = 1
TEST_SAMPLES = 20
RUN_NAME = time.strftime("%Y_%m_%d_%H_%M_%S") + '_GoL_delta_' + str(GOL_DELTA)
SNAPSHOTS_DIR = '../out/training/snapshots/{}'.format(RUN_NAME)
TENSORBOARD_LOGS_DIR = '../out/training/logs'
VIDEO_DIR = '../out/training/videos/{}'.format(RUN_NAME)


def state_step(state: np.array):
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
        state = state_step(state)

    return state


def create_training_sample(shape=(25, 25), warmup_steps=5, delta=GOL_DELTA):

    while True:
        start = create_random_board(shape, warmup_steps)
        end = start.copy()
        for i in range(delta):
            end = state_step(end)

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


def create_net_input_array(state: np.array, predicted_mask: np.array, predictions: np.array, outline_size=0):
    input_dead = (1 - state).astype(np.float)
    input_alive = state.astype(np.float)
    input_unpredicted = (1 - predicted_mask).astype(np.float)
    input_predicted_dead = ((1 - predictions) * predicted_mask).astype(np.float)
    input_predicted_alive = (predictions * predicted_mask).astype(np.float)
    sample_input = np.stack([
        input_dead,
        input_alive,
        input_unpredicted,
        input_predicted_dead,
        input_predicted_alive],
        axis=2
    )

    # plt.subplot(2, 1, 1)
    # plt.imshow(sample_input[:, :, 0])

    # outlining
    if outline_size > 0:
        tiles_y = ((outline_size - 1) // state.shape[0]) * 2 + 2 + 1
        tiles_x = ((outline_size - 1) // state.shape[1]) * 2 + 2 + 1
        offset_y = state.shape[0] - ((outline_size - 1) % state.shape[0]) - 1
        offset_x = state.shape[1] - ((outline_size - 1) % state.shape[1]) - 1

        sample_input = np.tile(sample_input, (tiles_y, tiles_x, 1))
        sample_input = sample_input[
                       offset_y:(offset_y + state.shape[0] + 2 * outline_size),
                       offset_x:(offset_x + state.shape[1] + 2 * outline_size)
                       ]

    # plt.subplot(2, 1, 2)
    # plt.imshow(sample_input[:, :, 0])
    # plt.show()

    return sample_input.transpose((2, 0, 1)).astype(np.float)


class GoLDataset(Dataset):
    def __init__(self, shape=(25, 25), warmup_steps=5, delta=GOL_DELTA, size=1024, outline_size=0):
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

        sample_input = create_net_input_array(end, predicted_mask, start, outline_size=self.outline_size)

        sample_target = start

        return {
            "input": sample_input,
            "mask": np.expand_dims(predicted_mask.astype(np.float), 3).transpose((2, 0, 1)),
            "target": sample_target
        }


class GoLModule(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            ResBlock(channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 2, 1)
        )

    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_l1_loss_of_parameters(self):
        loss = None
        for param in self.parameters():
            if loss is None:
                loss = torch.abs(param)
            else:
                loss += torch.abs(param)

        return loss * (1. / self.get_num_trainable_parameters())

    def get_l2_loss_of_parameters(self):
        loss = None
        for param in self.parameters():
            if loss is None:
                loss = torch.sum(torch.mul(param, param))
            else:
                loss += torch.sum(torch.mul(param, param))

        return loss * (1. / self.get_num_trainable_parameters())

    def forward(self, x):
        return self.main(x)

    def get_probabilities(self, x):
        return torch.softmax(self.main(x), 1)

    def get_best_guess(self, x: torch.tensor, mask: np.array):
        probabilities = self.get_probabilities(x)
        masked_probabilities = np.array(probabilities.tolist()) * (1 - mask)
        guess = np.unravel_index(masked_probabilities.argmax(), masked_probabilities.shape)
        return {
            "coord_yx": np.array([guess[2], guess[3]]),
            "alive": guess[1]
        }

    def get_tendencies_img(self, x):
        return np.array(self.get_probabilities(x).tolist()).transpose((0, 2, 3, 1))[0, :, :, 1]

    def solve(self,
              state: np.array,
              device: torch.device,
              ground_truth=None,
              plot_each_step=False,
              plot_result=False,
              video_fname=None):

        predicted_mask = np.zeros(state.shape)
        predictions = np.zeros(state.shape)

        total_runs = state.shape[0] * state.shape[1]

        video_out = None
        for i in range(total_runs):
            sample_input = create_net_input_array(state, predicted_mask, predictions, outline_size=4*2)
            batch_input = torch.from_numpy(np.expand_dims(sample_input, 0)).float().to(device)
            guess = self.get_best_guess(batch_input, predicted_mask)
            predicted_mask[guess['coord_yx'][0], guess['coord_yx'][1]] = 1
            predictions[guess['coord_yx'][0], guess['coord_yx'][1]] = guess['alive']

            if plot_each_step or (i == total_runs - 1 and plot_result) or video_fname is not None:

                # data
                fig = plt.figure(figsize=(16, 9), dpi=100)
                sub = fig.add_subplot(2, 3, 1)
                sub.set_title("start (ground truth)")
                if ground_truth is not None:
                    sub.imshow(ground_truth.astype(np.float))
                sub = fig.add_subplot(2, 3, 4)
                sub.set_title("end (input)")
                sub.imshow(state.astype(np.float))

                # net
                sub = fig.add_subplot(2, 3, 3)
                sub.set_title("Certainty heatmap")
                prob = self.get_tendencies_img(batch_input)
                overlay = np.ones((state.shape[0], state.shape[1], 4), dtype=np.float)
                overlay[:, :, 3] = predicted_mask
                # prob[prob < 0.5] *= -1
                # prob[prob < 0.5] += 1.0
                # prob *= (1 - prev_predicted_mask)
                sub.imshow(prob, vmin=0.0, vmax=1.0)
                sub.imshow(overlay, vmin=0.0, vmax=1.0)

                # outcome
                sub = fig.add_subplot(2, 3, 2)
                sub.set_title("net prediction")
                sub.imshow(predictions.astype(np.float))
                sub = fig.add_subplot(2, 3, 5)
                outc = predictions
                for d in range(GOL_DELTA):
                    outc = state_step(outc)
                sub.imshow(outc.astype(np.float))

                fig.canvas.draw()

                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                if video_fname is not None:
                    if video_out is None:

                        if not os.path.exists(os.path.dirname(video_fname)):
                            os.makedirs(os.path.dirname(video_fname))

                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_out = cv2.VideoWriter(video_fname, fourcc, 60.0,
                                                         (img.shape[1], img.shape[0]))
                    video_out.write(img[:, :, ::-1])
                    if i == total_runs - 1:
                        for n in range(59):
                            video_out.write(img[:, :, ::-1])
                plt.close(fig)

        if video_out is not None:
            video_out.release()

        return predictions


def main():
    device = torch.device('cuda')

    dataset = GoLDataset(size=STEPS_PER_EPOCH * BATCH_SIZE, outline_size=4*2)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=1,
                        pin_memory=True)

    net = GoLModule()
    net.to(device)
    display_sample = create_training_sample()
    test_samples = [create_training_sample() for i in range(TEST_SAMPLES)]
    optimizer = optim.Adam(net.parameters(), lr=LR)
    cel = nn.CrossEntropyLoss()

    train_writer = SummaryWriter(log_dir=TENSORBOARD_LOGS_DIR + '/' + RUN_NAME, comment=RUN_NAME)

    total_steps = 0
    for epoch in range(EPOCHS):
        batch_iter = iter(loader)
        for step in range(STEPS_PER_EPOCH):
            batch = next(batch_iter)

            net.zero_grad()
            torch_input = batch['input'].float().to(device)
            torch_target = batch['target'].long().to(device)
            prediction = net(torch_input)
            # weight_loss = net.get_l2_loss_of_parameters()
            classification_loss = cel(prediction, torch_target)
            loss = classification_loss  # weight_loss + classification_loss

            loss.backward()
            optimizer.step()

            if (step) % 16 == 0:
                loss_item = loss.item()
                print('Epoch {} - Step {} - loss {:.5f}'.format(epoch + 1, step + 1, loss_item))
                # train_writer.add_scalar('loss/weights-l2', weight_loss.item(), total_steps)
                train_writer.add_scalar('loss/class-ce', classification_loss.item(), total_steps)
                train_writer.add_scalar('loss/train', loss_item, total_steps)

            total_steps += 1

        net.eval()
        print("Create test video")
        net.solve(display_sample['end'],
                  device,
                  display_sample['start'],
                  video_fname='{}/epoch_{:03}.avi'.format(VIDEO_DIR, epoch + 1)
                  )

        print("Calculate epoch loss")
        epoch_loss = 0
        for test_sample in test_samples:
            res = net.solve(test_sample['end'], device)
            outc = res
            for d in range(GOL_DELTA):
                outc = state_step(outc)

            epoch_loss += np.mean(np.abs((outc - test_sample['end'])))
        epoch_loss /= len(test_samples)
        print("Epoch loss {}".format(epoch_loss))
        train_writer.add_scalar('loss/test', epoch_loss)
        net.train()

        if not os.path.isdir(SNAPSHOTS_DIR):
            os.makedirs(SNAPSHOTS_DIR)
        torch.save(net.state_dict(), '{}/epoch_{:03}.pt'.format(SNAPSHOTS_DIR, epoch + 1))

    print("Done!")


if __name__ == "__main__":
    main()
