import os
import sys
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from main import SUBMISSION_DIR, SUBMISSION_FILE_FORMAT, SCORE_FILE_FORMAT, create_training_sample, state_step

hashable_primes = np.array([
        2,     7,    23,    47,    61,     83,    131,    163,    173,    251,
      457,   491,   683,   877,   971,   2069,   2239,   2927,   3209,   3529,
     4451,  4703,  6379,  8501,  9293,  10891,  11587,  13457,  13487,  17117,
    18869, 23531, 23899, 25673, 31387,  31469,  36251,  42853,  51797,  72797,
    76667, 83059, 87671, 95911, 99767, 100801, 100931, 100937, 100987, 100999,
], dtype=np.int64)


def hash_geometric(board: np.ndarray) -> int:
    """
    Takes the 1D pixelwise view from each pixel (up, down, left, right) with wraparound
    the distance to each pixel is encoded as a prime number, the sum of these is the hash for each view direction
    the hash for each cell is the product of view directions and the hash of the board is the sum of these products
    this produces a geometric invariant hash that will be identical for roll / flip / rotate operations
    """
    assert board.shape[0] == board.shape[1]  # assumes square board
    size     = board.shape[0]
    l_primes = hashable_primes[:size//2+1]   # each distance index is represented by a different prime
    r_primes = l_primes[::-1]                # symmetric distance values in reversed direction from center

    hashed = 0
    for x in range(size):
        for y in range(size):
            # current pixel is moved to center [13] index
            horizontal = np.roll( board[:,y], size//2 - x)
            vertical   = np.roll( board[x,:], size//2 - y)
            left       = np.sum( horizontal[size//2:]   * l_primes )
            right      = np.sum( horizontal[:size//2+1] * r_primes )
            down       = np.sum( vertical[size//2:]     * l_primes )
            up         = np.sum( vertical[:size//2+1]   * r_primes )
            hashed    += left * right * down * up
    return hashed


if __name__ == "__main__":
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
    scores_values = np.array(df_scores.values).reshape((-1,))
    submission_values = np.array(df_submission.values)

    input_states = input_values[:, 2:].reshape((-1, 25 * 25))
    submission_states = submission_values[:, 1:].reshape((-1, 25 * 25))
    deltas = input_values[:, 1]

    print("start search")
    delta_1 = (deltas == 1).astype(np.int).reshape((-1, 1))
    delta_2 = (deltas == 2).astype(np.int).reshape((-1, 1))
    delta_3 = (deltas == 3).astype(np.int).reshape((-1, 1))
    delta_4 = (deltas == 4).astype(np.int).reshape((-1, 1))
    delta_5 = (deltas == 5).astype(np.int).reshape((-1, 1))
    print("calc mask")
    start_time = time.time()
    REPS = 50000
    for i in range(REPS):
        sample = create_training_sample(delta=1)

        deltas = [sample['start']]
        for d in range(5):
            deltas.append(state_step(deltas[-1]))

        deltas_states = np.array(list(reversed(deltas)), dtype=np.int).reshape((6, 1, 25*25))
        stop = deltas_states[0].reshape((1, 25 * 25,))

        new_scores = np.sum(np.abs(input_states - stop), axis=1)
        mask = np.clip(scores_values - new_scores, 0, 1)
        improvements = np.sum(mask)
        if improvements > 0:
            print("\nImprovements found: {}".format(improvements))
            mask = mask.reshape((-1, 1))
            inv_mask = 1 - mask
            submission_states = inv_mask * submission_states + \
                     delta_1 * mask * deltas_states[1] + \
                     delta_2 * mask * deltas_states[2] + \
                     delta_3 * mask * deltas_states[3] + \
                     delta_4 * mask * deltas_states[4] + \
                     delta_5 * mask * deltas_states[5]

            scores_values = mask.reshape((-1, )) * new_scores + inv_mask.reshape((-1, )) * scores_values

            print("Mean error is now {}".format(np.mean(scores_values, dtype=np.float) / (25 * 25)))

        print("{}/{} Estimated time left until finished: {} seconds".format(i+1, REPS, int((time.time() - start_time) * (REPS - i - 1) / (i + 1))))

    print("Rows left to check: {}".format(np.sum(scores_values > 0)))
    print("Writing files...")

    submission_values[:, 1:] = submission_states
    df_submission = pd.DataFrame(data=submission_values, columns=list(df_submission.columns.values))
    df_scores = pd.DataFrame(data=scores_values, columns=list(df_scores.columns.values))
    df_submission.to_csv(SUBMISSION_FILE_FORMAT.format(submission_id), index=False)
    df_scores.to_csv(SCORE_FILE_FORMAT.format(submission_id), index=False)

    print("Done!")
