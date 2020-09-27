import pandas as pd

if __name__ == "__main__":
    sub = pd.read_csv("../input/game-of-life/sub_1111.csv")
    sub.head()
    sub.to_csv("submission.csv", index=False)