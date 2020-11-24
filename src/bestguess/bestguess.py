import gc

import pandas as pd
import math

import cv2
import numpy as np
import matplotlib
import random

from tensorboard.backend.event_processing import event_accumulator
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import os

import torch.nn.functional as F

from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

TRAIN_DELTA = 1
MODEL_CHANNELS = 16 * 8
LR = 1e-4
BATCH_SIZE = 64         # normally 64
BATCHES_PER_STEP = 1
STEPS_PER_EPOCH = 2000
EPOCHS = 80
TEST_BATCHES = 16
TEST_SAMPLES_PER_BATCH = 16
TEST_SAMPLES = TEST_BATCHES * TEST_SAMPLES_PER_BATCH
HALF_LR_AFTER_N_EPOCHS = 7
RUN_NAME = str(MODEL_CHANNELS) + time.strftime("___%Y_%m_%d_%H_%M_%S") + '_sub_delta_' + str(TRAIN_DELTA)
SNAPSHOTS_DIR = '../../best_guess/out/training/snapshots/{}'.format(RUN_NAME)
TENSORBOARD_LOGS_DIR = '../../best_guess/out/training/logs'
VIDEO_DIR = '../../best_guess/out/training/videos/{}'.format(RUN_NAME)
SUBMISSION_DIR = '../../best_guess/out/submissions'
SUBMISSION_FILE_FORMAT = SUBMISSION_DIR + '/submission_{}.csv'
SCORE_FILE_FORMAT = SUBMISSION_DIR + '/score_{}.csv'


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


def state_loss(pred: np.array, target: np.array):
    return np.mean(np.abs(pred - target))


def plot(state: np.array):
    plt.imshow(state.astype(np.float))
    plt.show()


def create_random_board(shape=(25, 25), warmup_steps=5):
    factor = np.random.uniform(0.01, 0.99, (1, ))
    state = (np.random.uniform(0.0, 1.0, shape) > factor).astype(np.int)
    for i in range(warmup_steps):
        state = state_step(state)
    return state


def create_training_sample(shape=(25, 25), warmup_steps=5, delta=1, random_warmup=False):
    while True:
        start = create_random_board(shape, warmup_steps + (np.random.randint(0, 5) if random_warmup else 0))
        end = start.copy()
        for i in range(delta):
            end = state_step(end)
        if np.any(end):
            return {
                "start": start,
                "end": end,
                "delta": delta
            }


class TiledConv2d(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, [1, 1, 1, 1], mode='circular'))


class TiledResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            TiledConv2d(features, features),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            TiledConv2d(features, features)
        )

    def forward(self, x):
        return self.main(x) + x


def best_guess_block(input_channels, channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, channels, 1, bias=False),
        TiledResBlock(channels),
        TiledResBlock(channels),
        nn.BatchNorm2d(channels),
        nn.ReLU(True),
    )


def create_net_input_array(state: np.array, predicted_mask: np.array, predictions: np.array):
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

    return sample_input.transpose((2, 0, 1)).astype(np.float)


class BestGuessDataset(Dataset):
    def __init__(self, delta=1, shape=(25, 25), dataset_size=1024):
        self.delta = delta
        self.shape = shape
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sample = create_training_sample(self.shape, delta=self.delta)
        start = sample["start"]
        end = sample["end"]
        predicted_mask = (np.random.uniform(0.0, 1.0, self.shape) > np.random.uniform(0.0, 1.0, (1, ))).astype(np.int)

        # there needs to be a cell left to predict
        if np.sum(predicted_mask) == self.shape[0] * self.shape[1]:
            predicted_mask[np.random.randint(0, self.shape[0], (1, )), np.random.randint(0, self.shape[1], (1, ))] = 0

        return {
            "input": create_net_input_array(end, predicted_mask, start),
            "mask": np.expand_dims(predicted_mask.astype(np.float), 3).transpose((2, 0, 1)),
            "target": start
        }


class BestGuessModule(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.main = nn.Sequential(
            best_guess_block(5, channels),
            best_guess_block(channels, channels * 2),
            best_guess_block(channels * 2, channels * 4),
            nn.Conv2d(channels * 4, 2, 1)
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

    def get_best_guesses(self, x: torch.tensor, mask: np.array, num_guesses=2):
        probabilities = self.get_probabilities(x)
        masked_probabilities = np.array(probabilities.tolist()) * (1 - mask)
        guess = np.unravel_index(masked_probabilities.argmax(), masked_probabilities.shape)
        return {
            "coord_yx": np.array([guess[2], guess[3]]),
            "alive": guess[1]
        }

    def get_best_by_threshold(self, x: torch.tensor, mask: np.array, threshold: float):
        probabilities = self.get_probabilities(x)
        masked_probabilities = np.array(probabilities.tolist()) * (1 - mask)
        results = np.where(masked_probabilities >= threshold)
        return {
            "coord_yx": np.array([results[2], results[3]]),
            "alive": results[1]
        }

    def get_tendencies_img(self, x):
        return np.array(self.get_probabilities(x).tolist()).transpose((0, 2, 3, 1))[0, :, :, 1]

    def solve_pure_gpu(self, state: np.array, device: torch.device):
        predicted_mask = np.zeros(state.shape)
        predictions = np.zeros(state.shape)
        sample_input = create_net_input_array(state, predicted_mask, predictions)
        batch_input = torch.from_numpy(np.expand_dims(sample_input, 0)).float().to(device)

        for i in range(state.shape[0] * state.shape[1]):
            probabilities = torch.softmax(self.main(batch_input), 1)
            max_val = torch.argmax(probabilities.reshape((-1, 2 * state.shape[0] * state.shape[1])), 1, keepdim=True).reshape((-1, 2, state.shape[0], state.shape[1]))

            # batch_input[:, ]

    def solve_batch(self, states: np.array, device: torch.device):
        predicted_masks = np.zeros(states.shape)
        predictions = np.zeros(states.shape)
        batches_indices = np.arange(0, states.shape[0])

        total_runs = states.shape[1] * states.shape[2]
        sample_inputs = np.zeros((states.shape[0], 5, states.shape[1], states.shape[2]), dtype=np.float)

        for i in range(total_runs):
            for b in range(states.shape[0]):
                sample_inputs[b] = create_net_input_array(states[b], predicted_masks[b], predictions[b])

            input_tensor = torch.from_numpy(sample_inputs).float().to(device)
            probabilities = self.get_probabilities(input_tensor)
            masked_probabilities = np.array(probabilities.tolist()) * (1 - predicted_masks.reshape((-1, 1, states.shape[1], states.shape[2])))
            maxes = np.argmax(masked_probabilities.reshape((-1, 2 * states.shape[1] * states.shape[2])), axis=1)
            guesses = np.unravel_index(maxes, probabilities.shape)
            predicted_masks[batches_indices, guesses[2], guesses[3]] = 1
            predictions[batches_indices, guesses[2], guesses[3]] = guesses[1]

        return predictions

    def solve(self,
              state: np.array,
              device: torch.device,
              ground_truth=None,
              plot_each_step=False,
              plot_result=False,
              video_fname=None,
              quick_fill_threshold=1.0):

        predicted_mask = np.zeros(state.shape)
        predictions = np.zeros(state.shape)

        total_runs = state.shape[0] * state.shape[1]

        video_out = None
        for i in range(total_runs):
            sample_input = create_net_input_array(state, predicted_mask, predictions)
            batch_input = torch.from_numpy(np.expand_dims(sample_input, 0)).float().to(device)
            if quick_fill_threshold < 1:
                thr_guess = self.get_best_by_threshold(batch_input, predicted_mask, threshold=quick_fill_threshold)
                if len(thr_guess['alive']) > 0:
                    predicted_mask[thr_guess['coord_yx'][0], thr_guess['coord_yx'][1]] = 1
                    predictions[thr_guess['coord_yx'][0], thr_guess['coord_yx'][1]] = thr_guess['alive']
                else:
                    quick_fill_threshold = 1
            else:
                guess = self.get_best_guess(batch_input, predicted_mask)
                predicted_mask[guess['coord_yx'][0], guess['coord_yx'][1]] = 1
                predictions[guess['coord_yx'][0], guess['coord_yx'][1]] = guess['alive']

            if plot_each_step or (i == total_runs - 1 and plot_result) or video_fname is not None:

                # input
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
                overlay = np.ones((state.shape[0], state.shape[1], 4), dtype=np.float)
                overlay[:, :, 3] = (1.0 - predicted_mask) * 0.66
                sub.imshow(predictions.astype(np.float))
                sub.imshow(overlay, vmin=0.0, vmax=1.0)
                sub = fig.add_subplot(2, 3, 5)
                sub.set_title("prediction after {} steps".format(1))
                outc = predictions
                for d in range(1):
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


def create_plot_pdf(channels: np.ndarray, parameters: np.ndarray, means: np.ndarray, variances: np.ndarray):
    # plt.style.use('seaborn-whitegrid')
    f = plt.figure(figsize=(6, 4))
    plt.grid(color='gray')
    plt.plot(parameters, means, "-o")
    plt.xlabel('Anzahl Parameter in Mio.')
    plt.ylabel('Fehlerquote in Prozent')
    plt.fill_between(parameters, means - variances, means + variances,
                     color='gray', alpha=0.2)

    plt.show()
    f.savefig("losses_graph.pdf", bbox_inches='tight')


def test():
    TEST_DELTA = 2
    model_fnames = []
    log_fnames = []
    num_channels = []
    for fname in os.walk(os.path.dirname(SNAPSHOTS_DIR)):
        model_fname = fname[0] + '\epoch_033.pt'
        if '15_03_52_17' not in model_fname:
            continue
        if os.path.exists(model_fname):
            run_name = os.path.split(fname[0])[1]
            model_fnames.append(model_fname)
            num_channels.append(int(run_name.split('___')[0]))

            log_files = next(os.walk('{}/{}'.format(TENSORBOARD_LOGS_DIR, run_name)))
            log_fnames.append(os.path.join(log_files[0], log_files[2][0]))

    training_hours = []
    for log_fname in log_fnames:
        ea = event_accumulator.EventAccumulator(log_fname)
        ea.Reload()
        loss_train = ea.Scalars('loss/train')
        training_hours.append((loss_train[-1].wall_time - loss_train[0].wall_time) / 60 / 60)

    device = torch.device('cuda')
    num_test_batches = 50
    test_batch_size = 10
    end_states = np.array([create_training_sample(delta=5, random_warmup=False)['end'] for _ in range(num_test_batches * test_batch_size)]).reshape((num_test_batches, test_batch_size, 25, 25))
    means = []
    stds = []
    num_parameters = []
    for model_fname, channel_count in zip(model_fnames, num_channels):
        print('Testing {}'.format(model_fname))
        net = BestGuessModule(channel_count)
        net.load_state_dict(torch.load(model_fname))
        num_parameters.append(net.get_num_trainable_parameters())
        net.eval()
        net.to(device)

        losses = []
        for batch in end_states:
            predicted_starts = batch.copy()
            for g in range(TEST_DELTA):
                predicted_starts = net.solve_batch(predicted_starts, device)
            # predicted_starts = batch

            for pstop, gt in zip(predicted_starts, batch):
                for g in range(TEST_DELTA):
                    pstop = state_step(pstop)
                losses.append(state_loss(pstop, gt))
        losses = np.array(losses, dtype=np.float)
        print('Mean error in percent: {:.3f}%'.format(np.mean(losses) * 100))
        means.append(np.mean(losses))
        stds.append(np.std(losses))
        del net

    print('    \\begin{tabular}{rlllll}')
    print('        \\toprule')
    print('        ~ & ~ & ~ & ~ & \\multicolumn{2}{c}{Fehlerquote}\\\\')
    print('        \\cmidrule{5-6}')
    print('        Name & Channels & Parameter & Training & Durchsch. & Varianz\\\\')
    print('        \\midrule')
    for channel_count, mean, std, parameters, hours in zip(num_channels, means, stds, num_parameters, training_hours):
        print('        {} & ${}$ & ${:.1f}$ Mio. & ${:.1f}$ Std. & ${:.2f}\\%$ & ${:.2f}$ \\\\'.format(
            'bg{}'.format(channel_count),
            channel_count,
            parameters / 1000000,
            hours,
            mean * 100,
            std * 100
        ))
    print('        \\bottomrule')
    print('    \\end{tabular}')


def train():
    device = torch.device('cuda')

    dataset = BestGuessDataset(delta=TRAIN_DELTA, dataset_size=STEPS_PER_EPOCH * BATCH_SIZE * BATCHES_PER_STEP)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=5,
                        pin_memory=True)

    net = BestGuessModule(channels=MODEL_CHANNELS)
    # net.load_state_dict(torch.load('P:\\python\\convay-reversed\\best_guess\\out\\training\\snapshots\\128___2020_11_15_03_52_17_GoL_delta_1\\epoch_033.pt'))
    print('Num parameters: {}'.format(net.get_num_trainable_parameters()))
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    cel = nn.CrossEntropyLoss(reduction='none')

    train_writer = SummaryWriter(log_dir=TENSORBOARD_LOGS_DIR + '/' + RUN_NAME, comment=RUN_NAME)

    total_steps = 0
    for epoch in range(EPOCHS):
        batch_iter = iter(loader)
        for step in range(STEPS_PER_EPOCH):
            net.zero_grad()

            for sub_step in range(BATCHES_PER_STEP):
                batch = next(batch_iter)

                torch_input = batch['input'].float().to(device)
                torch_target = batch['target'].long().to(device)
                torch_mask = batch['mask'].float().to(device)
                prediction = net(torch_input)
                weight_loss = net.get_l2_loss_of_parameters()
                classification_loss = torch.mean(cel(prediction, torch_target) * torch_mask)
                loss = (weight_loss + classification_loss) * (1. / BATCHES_PER_STEP)

                loss.backward()
            optimizer.step()

            if step % 16 == 0:
                loss_item = loss.item()
                print('Epoch {} - Step {} - loss {:.5f}'.format(epoch + 1, step + 1, loss_item))
                train_writer.add_scalar('loss/weights-l2', weight_loss.item(), total_steps)
                train_writer.add_scalar('loss/class-ce', classification_loss.item(), total_steps)
                train_writer.add_scalar('loss/train', loss_item, total_steps)

            total_steps += 1

        net.eval()
        # print("Create test video")
        # display_sample = create_training_sample()
        # net.solve(display_sample['end'],
        #           device,
        #           display_sample['start'],
        #           video_fname='{}/epoch_{:03}.avi'.format(VIDEO_DIR, epoch + 1)
        #           )

        print("Calculate epoch loss")
        epoch_loss = 0
        for n in range(TEST_BATCHES):
            end_states = np.array([create_training_sample(delta=1, random_warmup=False)['end'] for _ in range(TEST_SAMPLES_PER_BATCH)])
            pred_start_states = net.solve_batch(end_states, device)
            for end_state, pred_start_state in zip(end_states, pred_start_states):
                pred_end_state = pred_start_state.copy()
                for g in range(TRAIN_DELTA):
                    pred_end_state = state_step(pred_end_state)
                epoch_loss += state_loss(end_state, pred_end_state)
        epoch_loss /= TEST_SAMPLES
        print("Epoch test loss {}".format(epoch_loss))
        train_writer.add_scalar('loss/test', epoch_loss, total_steps)
        net.train()

        # adjust lr
        new_lr = LR * math.pow(0.5, (epoch / HALF_LR_AFTER_N_EPOCHS))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        train_writer.add_scalar('lr', new_lr, total_steps)

        print("Save snapshot")
        if not os.path.isdir(SNAPSHOTS_DIR):
            os.makedirs(SNAPSHOTS_DIR)
        torch.save(net.state_dict(), '{}/epoch_{:03}.pt'.format(SNAPSHOTS_DIR, epoch + 1))

    print("Done!")


if __name__ == "__main__":
    train()
    # create_plot_pdf(
    #     np.array([16, 24, 32, 40, 48, 64, 80, 96]),
    #     np.array([0.2, 0.4, 0.8, 1.2, 1.8, 3.1, 4.9, 7.1]),
    #     np.array([1.31, 0.99, 0.85, 0.74, 0.69, 0.50, 0.44, 0.40]),
    #     np.array([1.23, 0.95, 0.90, 0.78, 0.76, 0.58, 0.54, 0.52])
    # )
    # test()
