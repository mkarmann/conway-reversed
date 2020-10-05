import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

matplotlib.use('TkAgg')


def binary_clamp(x: torch.tensor):
    return torch.clamp(x, 0, 1)


def conway_layer(x: torch.tensor):
    surround_sum = torch.roll(x, 1, 2) + torch.roll(x, -1, 2) + torch.roll(x, 1, 3) + torch.roll(x, -1, 3) +\
        torch.roll(x, (-1, -1), (2, 3)) + torch.roll(x, (1, -1), (2, 3)) + torch.roll(x, (-1, 1), (2, 3)) + torch.roll(x, (1, 1), (2, 3))
    return binary_clamp(surround_sum + x - 2) - binary_clamp(surround_sum - 3)


def tilePadding(x: torch.Tensor, left, right, top, bottom):
    padding = (left, right, top, bottom)

    top_left = np.array([top, left], dtype=np.int)
    bottom_right = np.array([bottom, right], dtype=np.int)
    shape = np.array((x.size()[2], x.size()[3]), dtype=np.int)

    tl_num_tiles = (top_left + (shape - 1)) // shape
    br_num_tiles = (top_left + (shape - 1)) // shape
    total_num_repeats = tl_num_tiles + br_num_tiles + 1

    tiled = x.repeat(1, 1, total_num_repeats[0], total_num_repeats[1])

    # cut out the wanted size
    out_shape = shape + top_left + bottom_right
    padding_top_left = tl_num_tiles * shape - top_left

    return tiled[:, :,
           padding_top_left[0]:(padding_top_left[0] + out_shape[0]),
           padding_top_left[1]:(padding_top_left[1] + out_shape[1])
           ]


class TilePad2d(nn.Module):
    def __init__(self, left, right, top, bottom):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def forward(self, x):
        return tilePadding(x, self.left, self.right, self.top, self.bottom)


class BestChangeLayer(nn.Module):
    def __init__(self, delta=1, window=(3, 3), device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.window = window
        self.influence_window = (window[0] + 4 * delta, window[1] + 4 * delta)
        self.delta = delta
        self.num_bins = window[0] * window[1]
        self.num_possible_window_inputs = 2 ** self.num_bins
        self.possible_inputs = np.zeros((self.num_possible_window_inputs, self.num_bins))

        # compute all possible
        for i in range(self.num_possible_window_inputs):
            self.possible_inputs[i] = np.array(list(np.binary_repr(i, self.num_bins)), dtype=np.float)
        self.possible_inputs = self.possible_inputs.reshape((1, self.num_possible_window_inputs, window[0], window[1]))

        self.possible_inputs_mask = np.zeros((1, self.num_possible_window_inputs, self.influence_window[0], self.influence_window[1]), dtype=np.float)
        self.possible_inputs_mask[:, :, 2*delta:-2*delta, 2*delta:-2*delta] = 1
        self.possible_inputs_values = np.zeros((1, self.num_possible_window_inputs, self.influence_window[0], self.influence_window[1]), dtype=np.float)
        self.possible_inputs_values[:, :, 2*delta:-2*delta, 2*delta:-2*delta] = self.possible_inputs

        self.pi = torch.from_numpy(self.possible_inputs).float().to(device)
        self.pi_window = torch.from_numpy(self.possible_inputs_values).float().to(device)
        self.pi_window_mask = torch.from_numpy(self.possible_inputs_mask).float().to(device)
        self.pi_window_inv_mask = -self.pi_window_mask + 1

        self.replication_input_layer = TilePad2d(
            delta * 2,
            delta * 2 + window[1] - 1,
            delta * 2,
            delta * 2 + window[0] - 1)

        self.replication_target_layer = TilePad2d(
            delta,
            delta + window[1] - 1,
            delta,
            delta + window[0] - 1)

        self.unpool = nn.MaxUnpool1d(self.num_possible_window_inputs)

    def forward(self, x: torch.Tensor, target: torch.Tensor):

        random_x = np.random.randint(0, x.size()[3])
        random_y = np.random.randint(0, x.size()[2])

        influence_window = self.replication_input_layer(x)[:, :, random_y:(random_y + self.influence_window[0]), random_x:(random_x + self.influence_window[1])]
        target_window = self.replication_target_layer(target)[:, :, random_y:(random_y + self.window[0] + 2 * self.delta), random_x:(random_x + self.window[1] + 2 * self.delta)]
        process_window = influence_window.repeat(1, self.num_possible_window_inputs, 1, 1)
        process_window = process_window * self.pi_window_inv_mask + self.pi_window
        end = process_window
        for d in range(self.delta):
            end = conway_layer(end)[:, :, 1:-1, 1:-1]

        errors = torch.sum(torch.abs(end - target_window.repeat(1, self.num_possible_window_inputs, 1, 1)), (2, 3))
        seeded_errors = errors + torch.rand(errors.size(), device=self.device) * 0.5

        indices = torch.argmin(seeded_errors, 1)
        best_mask = self.unpool(torch.ones((seeded_errors.size()[0], 1, 1), device=self.device), indices.reshape(-1, 1, 1)).reshape(-1, self.num_possible_window_inputs, 1, 1)
        best_inputs = torch.sum(self.pi * best_mask, 1).reshape((-1, 1, self.window[0], self.window[1]))

        out = torch.roll(x, shifts=(-random_y, -random_x), dims=(2, 3))
        out[:, :, :self.window[0], :self.window[1]] = best_inputs
        out = torch.roll(out, shifts=(random_y, random_x), dims=(2, 3))

        return out

    def solve_batch(self, states: np.array, device: torch.device, num_steps=1000):
        target_batch = torch.from_numpy(states).reshape(states.shape[0], 1, states.shape[1], states.shape[2]).float().to(device)
        inputs_batch = torch.sign(torch.rand(target_batch.size(), device=device).float() - 0.5) * 0.5 + 0.5

        for i in range(num_steps):
            inputs_batch = self(inputs_batch, target_batch)

        return np.clip(np.rint(np.array(inputs_batch.tolist())).astype(np.int), 0, 1).reshape(states.shape)


def main():
    DELTA = 1
    STEPS = 2000

    device = torch.device('cuda')

    sample = create_training_sample(delta=DELTA)
    batch_target = np.expand_dims(np.expand_dims(sample['end'], 0), 3)
    batch_sample_start = np.expand_dims(np.expand_dims(sample['start'], 0), 3)
    batch_target_torch = torch.from_numpy(batch_target.transpose((0, 3, 1, 2))).float().to(device)
    batch_sample_start_torch = torch.from_numpy(batch_sample_start.transpose((0, 3, 1, 2))).float().to(device)

    # analytic_start_state = torch.zeros(batch_target_torch.size()).float()
    analytic_start_state = torch.sign(torch.rand(batch_target_torch.size(), device=device).float() - 0.5) * 0.5 + 0.5
    # analytic_start_state = batch_target_torch
    # plot(sample['start'])
    # plot(sample['end'])

    bcl = BestChangeLayer(delta=DELTA, device=device)
    loss = torch.sum(torch.abs(batch_target_torch - conway_layer(analytic_start_state)))
    print(loss.item())
    for step in range(STEPS):
        analytic_start_state = bcl(analytic_start_state, batch_target_torch)
        predicted_end = analytic_start_state
        for d in range(DELTA):
            predicted_end = conway_layer(predicted_end)
        loss = torch.sum(torch.abs(batch_target_torch - predicted_end))
        print("errors: {} - loss {}".format(loss.item(), loss.item() / (25 * 25)))


if __name__ == "__main__":
    main()
