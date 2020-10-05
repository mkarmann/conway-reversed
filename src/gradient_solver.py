import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

from main import create_training_sample, plot, state_loss, state_step

matplotlib.use('TkAgg')


def is_positive(x: torch.tensor):
    return torch.clamp(x, 0, 1)


def conway_layer(x: torch.tensor):
    surround_sum = torch.roll(x, 1, 2) + torch.roll(x, -1, 2) + torch.roll(x, 1, 3) + torch.roll(x, -1, 3) +\
        torch.roll(x, (-1, -1), (2, 3)) + torch.roll(x, (1, -1), (2, 3)) + torch.roll(x, (-1, 1), (2, 3)) + torch.roll(x, (1, 1), (2, 3))

    return is_positive(surround_sum + x - 2) - is_positive(surround_sum - 3)


def custom_gradient_conway_loss(x0: torch.Tensor, target: torch.Tensor, delta: int):
    """

    input roll pattern:

    00      01      02


    10      11      12


    20      21      22

    """
    x0_21 = torch.roll(x0, 1, 2)
    x0_01 = torch.roll(x0, -1, 2)
    x0_12 = torch.roll(x0, 1, 3)
    x0_10 = torch.roll(x0, -1, 3)
    x0_00 = torch.roll(x0, (-1, -1), (2, 3))
    x0_20 = torch.roll(x0, (1, -1), (2, 3))
    x0_02 = torch.roll(x0, (-1, 1), (2, 3))
    x0_22 = torch.roll(x0, (1, 1), (2, 3))

    # step one delta
    surround_sum = x0_21 + x0_01 + x0_12 + x0_10 + x0_00 + x0_20 + x0_02 + x0_22
    x1 = is_positive(surround_sum + x0 - 2) - is_positive(surround_sum - 3)

    # step multiple deltas
    if delta > 1:
        end, gradient1, loss = custom_gradient_conway_loss(x1, target, delta - 1)
    else:
        gradient1 = target - x1

    # masks
    mask_x0_1 = x0
    mask_x0_0 = -x0 + 1
    pos_gradient1 = torch.clamp_min(gradient1, 0)
    neg_gradient1 = torch.clamp_max(gradient1, 0)
    mask_surround_sum_is_2 = is_positive(surround_sum - 1) - is_positive(surround_sum - 2)
    mask_surround_sum_below_3 = is_positive(-surround_sum + 3)
    mask_surround_sum_above_3 = is_positive(surround_sum - 3)

    # randomness (-1 or 1)
    random_gradient_direction = torch.sign(torch.rand(x0.size()) - 1.1)

    # ------------------
    # calculate gradient
    # ------------------

    # gradient negative
    center_gradient = neg_gradient1 * mask_surround_sum_is_2
    surround_gradient = neg_gradient1 * random_gradient_direction

    # gradient positive - below 3
    center_gradient += pos_gradient1 * mask_surround_sum_below_3
    surround_gradient += pos_gradient1 * mask_surround_sum_below_3

    # gradient positive - above 3
    surround_gradient -= pos_gradient1 * mask_surround_sum_above_3

    # sum up the gradients
    sg_21 = torch.roll(surround_gradient, 1, 2)
    sg_01 = torch.roll(surround_gradient, -1, 2)
    sg_12 = torch.roll(surround_gradient, 1, 3)
    sg_10 = torch.roll(surround_gradient, -1, 3)
    sg_00 = torch.roll(surround_gradient, (-1, -1), (2, 3))
    sg_20 = torch.roll(surround_gradient, (1, -1), (2, 3))
    sg_02 = torch.roll(surround_gradient, (-1, 1), (2, 3))
    sg_22 = torch.roll(surround_gradient, (1, 1), (2, 3))
    gradient0 = center_gradient + sg_21 + sg_01 + sg_12 + sg_10 + sg_00 + sg_20 + sg_02 + sg_22

    # keep only valid direction gradients
    gradient0 = mask_x0_0 * torch.clamp_min(gradient0, 0) + mask_x0_1 * torch.clamp_max(gradient0, 0)

    if delta > 1:
        return end, gradient0, loss
    else:
        return x1, gradient0, torch.sum(torch.abs(gradient1))


class ConvRegressionModule(nn.Module):
    def __init__(self, input_shape=(1, 1, 25, 25)):
        super().__init__()
        self.inputs = torch.from_numpy(np.random.uniform(0.4, 0.6, input_shape)).float()
        self.inputs_para = nn.Parameter(self.inputs, True)
        self.register_parameter(name="bias", param=nn.Parameter(self.inputs, True))

    def forward(self, num_iterations):
        out = conway_layer(self.inputs_para)
        for i in range(num_iterations - 1):
            out = conway_layer(out)

        return out

    def get_current_boards(self):
        out = (np.array(self.inputs.tolist()) >= 0.5).astype(np.int)
        return out[:, 0, :, :]

    def step_clamp(self):
        self.inputs.data.clamp(0, 1)


def plot_solver_state(input_mask: torch.Tensor,
                      target: torch.Tensor,
                      analytic_start_state: torch.Tensor,
                      output: torch.Tensor,
                      gradient: torch.Tensor):
    input_mask = np.array(input_mask.tolist())[0, 0]
    target = np.array(target.tolist())[0, 0]
    analytic_start_state = np.array(analytic_start_state.tolist())[0, 0]
    output = np.array(output.tolist())[0, 0]
    gradient = np.array(gradient.tolist())[0, 0]

    plt.clf()
    plt.subplot(2, 3, 1)
    plt.title("input")
    plt.imshow(input_mask, vmin=0.0, vmax=1.0)

    plt.subplot(2, 3, 2)
    plt.title("gradients")
    plt.imshow(gradient)

    plt.subplot(2, 3, 3)
    plt.title("input (analytic)")
    plt.imshow(analytic_start_state, vmin=0.0, vmax=1.0)

    plt.subplot(2, 2, 3)
    plt.title("target")
    plt.imshow(target, vmin=0.0, vmax=1.0)

    plt.subplot(2, 2, 4)
    plt.title("current output")
    plt.imshow(output, vmin=0.0, vmax=1.0)
    plt.show(block=False)


def main():
    DELTA = 5
    STEPS = 700

    sample = create_training_sample(delta=DELTA)
    batch_target = np.expand_dims(np.expand_dims(sample['end'], 0), 3)
    batch_sample_start = np.expand_dims(np.expand_dims(sample['end'], 0), 3)
    batch_target_torch = torch.from_numpy(batch_target.transpose((0, 3, 1, 2))).float()
    batch_sample_start_torch = torch.from_numpy(batch_sample_start.transpose((0, 3, 1, 2))).float()

    analytic_start_state = torch.rand(batch_sample_start_torch.size()) * 0.1
    # analytic_start_state = batch_target_torch
    # plot(sample['start'])
    # plot(sample['end'])

    max_pool = nn.MaxPool2d((25, 25), return_indices=True)
    max_unpool = nn.MaxUnpool2d((25, 25))
    unpool_value = torch.ones((1, 1, 1, 1)).float()

    for step in range(STEPS):
        masked = torch.sign(analytic_start_state - 0.5) * 0.5 + 0.5

        output, gradient, loss = custom_gradient_conway_loss(masked, batch_target_torch, DELTA)
        print(loss.item())
        plot_solver_state(masked, batch_target_torch, analytic_start_state, output, gradient)

        max, indices = max_pool(torch.abs(gradient))
        biggest_gradient = torch.sign(gradient * max_unpool(unpool_value, indices))
        analytic_start_state += biggest_gradient

        # analytic_start_state = torch.clamp(analytic_start_state + gradient * 3 / (torch.sum(torch.abs(gradient)) + 0.00001), 0., 1.0)
        plt.pause(0.01)

    print()

    exit()

    net = ConvRegressionModule(batch_target_torch.size())
    mse = torch.nn.MSELoss(reduction="sum")
    sgd = torch.optim.SGD(net.parameters(), 0.01)

    prediction = net.get_current_boards()
    pred_end = np.array(prediction.tolist())[0]
    plt.imshow(pred_end)
    plt.show()
    for step in range(STEPS):
        net.zero_grad()
        loss = mse(net(DELTA), batch_target_torch)
        loss.backward()
        print(loss.item())
        sgd.step()
        net.step_clamp()

    prediction = net.get_current_boards()
    pred_end = np.array(prediction.tolist())[0]
    for d in range(DELTA):
        pred_end = state_step(pred_end)

    print(state_loss(sample['end'], pred_end) * 25 * 25)


if __name__ == "__main__":
    main()
