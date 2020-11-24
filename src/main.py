import pandas as pd
import math

import cv2
import numpy as np
import matplotlib
import random

from bestguess.bestguess import BestGuessModule, MODEL_CHANNELS
from stochastic_optimizer import BestChangeLayer

# matplotlib.use('agg')
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
EPOCHS = 128
GOL_DELTA = 2
TEST_SAMPLES = 20
HALF_LR_AFTER_N_EPOCHS = 32
OUTLINE_SIZE = 5*2
RUN_NAME = time.strftime("%Y_%m_%d_%H_%M_%S") + '_GoL_delta_' + str(GOL_DELTA)
SNAPSHOTS_DIR = '../out/training/snapshots/{}'.format(RUN_NAME)
TENSORBOARD_LOGS_DIR = '../out/training/logs'
VIDEO_DIR = '../out/training/videos/{}'.format(RUN_NAME)
SUBMISSION_DIR = '../out/submissions'
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


def create_training_sample(shape=(25, 25), warmup_steps=5, delta=GOL_DELTA, random_warmup=False):

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


class ResBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.main = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features // 4, 3, padding=0, bias=False),
            nn.BatchNorm2d(features // 4),
            nn.ReLU(True),
            nn.Conv2d(features // 4, features, 1, padding=0,  bias=False)
        )

    def forward(self, x):
        return self.main(x) + x[:, :, 1:-1, 1:-1]


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
        sample = create_training_sample(self.shape, self.warmup_steps, self.delta, random_warmup=True)
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
            ResBlock(channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            ResBlock(channels),
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
        sample_input = create_net_input_array(state, predicted_mask, predictions, outline_size=OUTLINE_SIZE)
        batch_input = torch.from_numpy(np.expand_dims(sample_input, 0)).float().to(device)

        for i in range(state.shape[0] * state.shape[1]):
            probabilities = torch.softmax(self.main(batch_input), 1)
            max_val = torch.argmax(probabilities.reshape((-1, 2 * state.shape[0] * state.shape[1])), 1, keepdim=True).reshape((-1, 2, state.shape[0], state.shape[1]))

            # batch_input[:, ]

    def solve_batch(self, states: np.array, device: torch.device, best_guesses_per_sample=1):
        predicted_masks = np.zeros(states.shape)
        predictions = np.zeros(states.shape)
        batches_indices = np.arange(0, states.shape[0])

        total_runs = states.shape[1] * states.shape[2]
        sample_inputs = np.zeros((states.shape[0], 5, states.shape[1] + 2 * OUTLINE_SIZE, states.shape[2] + 2 * OUTLINE_SIZE), dtype=np.float)

        for i in range(total_runs):
            for b in range(states.shape[0]):
                sample_inputs[b] = create_net_input_array(states[b], predicted_masks[b], predictions[b], outline_size=OUTLINE_SIZE)
            input_tensor = torch.from_numpy(sample_inputs).float().to(device)


            if best_guesses_per_sample == 1 or i > total_runs // 2:

                # only chose the best guess
                probabilities = self.get_probabilities(input_tensor)
                masked_probabilities = np.array(probabilities.tolist()) * (1 - predicted_masks.reshape((-1, 1, states.shape[1], states.shape[2])))
                maxes = np.argmax(masked_probabilities.reshape((-1, 2 * states.shape[1] * states.shape[2])), axis=1)
                guesses = np.unravel_index(maxes, probabilities.shape)
                predicted_masks[batches_indices, guesses[2], guesses[3]] = 1
                predictions[batches_indices, guesses[2], guesses[3]] = guesses[1]

            else:

                num_best_guesses = min(best_guesses_per_sample, total_runs - i)

                # select between multiple best guesses
                probabilities = self.get_probabilities(input_tensor)
                masked_probabilities = np.array(probabilities.tolist()) * (
                            1 - predicted_masks.reshape((-1, 1, states.shape[1], states.shape[2])))
                maxes_all = np.zeros((states.shape[0], num_best_guesses), dtype=np.int)
                mkp = masked_probabilities.reshape((-1, 2 * states.shape[1] * states.shape[2]))
                for g in range(num_best_guesses):
                    maxes = np.argmax(mkp, axis=1)
                    maxes_all[:, g] = maxes
                    mkp[batches_indices, maxes] = 0

                maxes = maxes_all[:, np.random.randint(0, num_best_guesses, states.shape[0])]
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
            sample_input = create_net_input_array(state, predicted_mask, predictions, outline_size=OUTLINE_SIZE)
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
                sub.set_title("prediction after {} steps".format(GOL_DELTA))
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


def test():
    device = torch.device('cuda')
    net = GoLModule()
    net.load_state_dict(torch.load('../out/training/snapshots/2020_09_25_18_51_41_GoL_delta_1/epoch_103.pt'))
    net.eval()
    net.to(device)

    df_input = pd.read_csv("../input/test.csv")
    input_values = df_input.values

    error_sum = 0
    for i in range(100):
        delta = input_values[i][1]
        end_state = input_values[i][2:].reshape((25, 25))
        gt_states = [end_state]
        for d in range(delta):
            gt_states.append(state_step(gt_states[-1]))

        current_solve = end_state
        for d in range(delta):
            prev_solve = current_solve.copy()
            current_solve = net.solve(current_solve, device)      # video_fname='out/vid_out_{}.avi'.format(d + 1))

        # check error
        outc = current_solve
        for d in range(delta):
            outc = state_step(outc)
        c_error = state_loss(outc, end_state)
        print('Error d{}: {}'.format(delta, c_error))

        error_sum += c_error
        print("Moving avg: {}\n".format(error_sum / (i + 1)))
    print('\nMean Error: {}'.format(error_sum / 100))


def improve_submission():

    DELTA = 5

    if not os.path.isdir(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)

    submission_id = 0
    while os.path.exists(SUBMISSION_FILE_FORMAT.format(submission_id)):
        submission_id += 1

    if submission_id == 0:
        df_submission = pd.read_csv("../input/sample_submission.csv")
        df_scores = pd.DataFrame(np.ones(len(df_submission), dtype=np.int) * 25 * 25, columns=["num_wrong_cells"])
    else:
        df_submission = pd.read_csv(SUBMISSION_FILE_FORMAT.format(submission_id - 1))
        df_scores = pd.read_csv(SCORE_FILE_FORMAT.format(submission_id - 1))

    input_csv = "../input/test.csv"
    df_input = pd.read_csv(input_csv)
    input_values = np.array(df_input.values)
    scores_values = np.array(df_scores.values)
    submission_values = np.array(df_submission.values)

    to_check_rows = np.sum(scores_values > 0)
    print("To check rows: {}".format(to_check_rows))

    # -------------------
    # init models
    # -------------------

    device = torch.device('cuda')
    # net = GoLModule()
    # net.load_state_dict(torch.load('../out/training/snapshots/2020_09_25_18_51_41_GoL_delta_1/epoch_103.pt'))
    # net.eval()
    # net.to(device)
    net = BestGuessModule(channels=MODEL_CHANNELS)
    net.load_state_dict(torch.load('P:\\python\\convay-reversed\\best_guess\\out\\training\\snapshots\\128___2020_11_15_03_52_17_GoL_delta_1\\epoch_033.pt'))
    print('Num parameters: {}'.format(net.get_num_trainable_parameters()))
    net.to(device)
    net.eval()
    bcls = [
        BestChangeLayer(window=(3, 3), delta=DELTA, device=device)
    ]

    # -------------------

    # loop through examples
    start_time = time.time()
    batch_size = 16
    stop_states = np.zeros((batch_size, 25, 25), dtype=np.int)
    indices = np.zeros(batch_size, dtype=np.int)
    current_sample_idx = 0
    for i in range(len(submission_values)):

        # ignore if errors are already zero
        old_score = scores_values[i][0]
        if old_score == 0:
            continue

        delta = input_values[i][1]
        stop_state = input_values[i][2:].reshape((25, 25))

        # skipping if wanted
        if delta != DELTA:
            continue

        indices[current_sample_idx] = i
        stop_states[current_sample_idx] = stop_state
        current_sample_idx = (current_sample_idx + 1) % batch_size
        if current_sample_idx != 0:
            continue

        print("\nRow {} with delta {} has old score: {}".format(i + 1, delta, old_score))

        # ------------------------
        # do the single prediction
        # ------------------------
        pred_start_states = stop_states.copy()
        bcl = random.choice(bcls)
        for d in range(delta):
            pred_start_states = net.solve_batch(pred_start_states, device)
        pred_start_states = bcl.solve_batch(stop_states, device, num_steps=4000, initial_states=pred_start_states)

        # -------------------

        for index, pred_start_state, stop_state in zip(indices, pred_start_states, stop_states):
            pred_end_state = pred_start_state.copy()
            for d in range(delta):
                pred_end_state = state_step(pred_end_state)

            new_score = np.sum(np.abs(pred_end_state - stop_state))
            old_score = scores_values[index][0]
            if new_score < old_score:
                print("Improved from {} to {}!".format(old_score, new_score))
                submission_values[index, 1:] = pred_start_state.reshape((-1,))
                scores_values[index] = new_score
                print("Mean error is now {}".format(np.mean(scores_values, dtype=np.float) / (25 * 25)))

        print("Estimated time left until finished: {} seconds".format(int((time.time() - start_time) * (len(submission_values) - i - 1) / (i + 1))))

    print("\n------------------------")
    print("Mean error is now {}".format(np.mean(scores_values) / (25 * 25)))
    print("Rows left to check: {}".format(np.sum(scores_values > 0)))
    print("Writing files...")

    df_submission = pd.DataFrame(data=submission_values, columns=list(df_submission.columns.values))
    df_scores = pd.DataFrame(data=scores_values, columns=list(df_scores.columns.values))
    df_submission.to_csv(SUBMISSION_FILE_FORMAT.format(submission_id), index=False)
    df_scores.to_csv(SCORE_FILE_FORMAT.format(submission_id), index=False)

    print("Done!")


def train():
    device = torch.device('cuda')

    dataset = GoLDataset(size=STEPS_PER_EPOCH * BATCH_SIZE, outline_size=OUTLINE_SIZE)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        num_workers=1,
                        pin_memory=True)

    net = GoLModule()
    print('Num parameters: {}'.format(net.get_num_trainable_parameters()))
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    cel = nn.CrossEntropyLoss(reduction='none')

    train_writer = SummaryWriter(log_dir=TENSORBOARD_LOGS_DIR + '/' + RUN_NAME, comment=RUN_NAME)

    total_steps = 0
    for epoch in range(EPOCHS):
        batch_iter = iter(loader)
        for step in range(STEPS_PER_EPOCH):
            batch = next(batch_iter)

            net.zero_grad()
            torch_input = batch['input'].float().to(device)
            torch_target = batch['target'].long().to(device)
            torch_mask = batch['mask'].float().to(device)
            prediction = net(torch_input)
            # weight_loss = net.get_l2_loss_of_parameters()
            classification_loss = torch.mean(cel(prediction, torch_target) * torch_mask)
            loss = classification_loss  # weight_loss + classification_loss

            loss.backward()
            optimizer.step()

            if step % 16 == 0:
                loss_item = loss.item()
                print('Epoch {} - Step {} - loss {:.5f}'.format(epoch + 1, step + 1, loss_item))
                # train_writer.add_scalar('loss/weights-l2', weight_loss.item(), total_steps)
                train_writer.add_scalar('loss/class-ce', classification_loss.item(), total_steps)
                train_writer.add_scalar('loss/train', loss_item, total_steps)

            total_steps += 1

        # print("Create test video")
        net.eval()
        # display_sample = create_training_sample()
        # net.solve(display_sample['end'],
        #           device,
        #           display_sample['start'],
        #           video_fname='{}/epoch_{:03}.avi'.format(VIDEO_DIR, epoch + 1)
        #           )

        print("Calculate epoch loss")
        epoch_loss = 0
        for t in range(TEST_SAMPLES):
            test_sample = create_training_sample(random_warmup=True)
            res = net.solve(test_sample['end'], device)
            outc = res
            for d in range(GOL_DELTA):
                outc = state_step(outc)
            epoch_loss += np.mean(np.abs((outc - test_sample['end'])))
        epoch_loss /= TEST_SAMPLES
        print("Epoch loss {}".format(epoch_loss))
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
    improve_submission()
