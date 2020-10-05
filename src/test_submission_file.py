import numpy as np
import pandas as pd
from main import state_step, state_loss
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SUBMISSION_FILE = "../out/submissions/submission_9.csv"


def main():
    input_csv = "../input/test.csv"

    df_input = pd.read_csv(input_csv)
    df_submission = pd.read_csv(SUBMISSION_FILE)

    print(df_input.head())
    print(df_submission.head())

    input_values = df_input.values
    submission_values = df_submission.values

    losses = np.zeros((len(input_values),), dtype=np.float)
    sum = 0
    for i in range(len(input_values)):
        delta = input_values[i][1]
        stop_state = input_values[i][2:].reshape((25, 25))
        pred_start_state = submission_values[i][1:].reshape((25, 25))

        pred_end_state = pred_start_state.copy()
        for d in range(delta):
            pred_end_state = state_step(pred_end_state)

        loss = state_loss(pred_end_state, stop_state)
        sum += loss
        losses[i] = loss

    print("mean error: {}".format(sum / len(input_values)))

    kwargs = dict(alpha=0.5, bins=20)

    deltas = input_values[:, 1]
    colors = ['b', 'g', 'r', 'c', 'm']
    for i in range(5):
        plt.hist(losses[deltas == i + 1], **kwargs, color=colors[i], label='delta {}'.format(i + 1))
    plt.gca().set(title='Distribution of errors per delta', ylabel='Frequency', xlabel='loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
