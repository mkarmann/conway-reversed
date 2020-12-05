import os

import numpy as np
import pandas as pd

from main import SUBMISSION_DIR, SUBMISSION_FILE_FORMAT, SCORE_FILE_FORMAT, state_step, plot

# TO_FUSE_SUBMISSION_FNAME = "../data/imgsegsolver/submission.csv"
# TO_FUSE_SUBMISSION_FNAME = "../data/z3solver/submission.csv"
# TO_FUSE_SUBMISSION_FNAME = "../data/SoliSet/submission.csv" # no change at all
# TO_FUSE_SUBMISSION_FNAME = "../data/merge/submission.csv"
TO_FUSE_SUBMISSION_FNAME = "../data/kandm/submission.csv"


def improve_submission():
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
    input_values = df_input.values
    scores_values = np.array(df_scores.values)
    submission_values = np.array(df_submission.values)

    df_fuse_submission = pd.read_csv(TO_FUSE_SUBMISSION_FNAME)
    fuse_submission_values = df_fuse_submission.values

    for i in range(len(submission_values)):

        # ignore if errors are already zero
        old_score = scores_values[i][0]
        if old_score == 0:
            continue

        delta = input_values[i][1]
        stop_state = input_values[i][2:].reshape((25, 25))
        start_state = submission_values[i][1:].reshape((25, 25))
        fuse_start_state = fuse_submission_values[i][1:].reshape((25, 25))

        pred_end_state = fuse_start_state.copy()
        for d in range(delta):
            pred_end_state = state_step(pred_end_state)

        new_score = np.sum(np.abs(pred_end_state - stop_state))
        old_score = scores_values[i][0]
        if new_score < old_score:
            print("Improved from {} to {}!".format(old_score, new_score))
            submission_values[i, 1:] = fuse_start_state.reshape((-1,))
            scores_values[i] = new_score
            print("Mean error is now {}".format(np.mean(scores_values, dtype=np.float) / (25 * 25)))

    print("\n------------------------")
    print("Mean error is now {}".format(np.mean(scores_values) / (25 * 25)))
    print("Rows left to check: {}".format(np.sum(scores_values > 0)))
    print("Writing files...")

    df_submission = pd.DataFrame(data=submission_values, columns=list(df_submission.columns.values))
    df_scores = pd.DataFrame(data=scores_values, columns=list(df_scores.columns.values))
    df_submission.to_csv(SUBMISSION_FILE_FORMAT.format(submission_id), index=False)
    df_scores.to_csv(SCORE_FILE_FORMAT.format(submission_id), index=False)

    print("Done!")


if __name__ == '__main__':
    improve_submission()
