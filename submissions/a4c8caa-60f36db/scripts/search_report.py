import glob
import sys
import os
import pandas as pd


if len(sys.argv) == 1:
    print(f"usage: {sys.argv[0]} logdir hparams_dir out_path")
    sys.exit(1)

_, logdir, hparams_dir, out_path = sys.argv

metrics = pd.concat([
    pd.read_csv(f, index_col="version") for f in glob.glob(os.path.join(logdir, "*", "*", "best_metrics.csv"))
])

hyperparams = pd.concat([
    pd.read_csv(f, index_col="version") for f in glob.glob(os.path.join(hparams_dir, "hparams*.csv"))
])

data = pd.concat((metrics, hyperparams), join="inner", axis=1)
data.to_csv(os.path.join(out_path, "metrics.csv"))
