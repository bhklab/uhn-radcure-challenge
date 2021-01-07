import os
import subprocess
import tempfile
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import ParameterSampler
from yaml import load, Loader
from scipy.stats import uniform, loguniform


def parse_config(config_file):
    with open(config_file, "r") as f:
        config = load(f, Loader=Loader)
    sbatch_args = []
    for k, v in config["sbatch"].items():
        if len(k) == 1:
            sbatch_args.append(f"-{k} {v}")
        else:
            sbatch_args.append(f"--{k}={v}")
    param_dists = {}
    for k, v in config["hyperparams"].items():
        if isinstance(v, dict):
            lo, hi = v["range"]
            if v["dist"] == "uniform":
                param_dists[k] = uniform(lo, hi)
            elif v["dist"] == "loguniform":
                param_dists[k] = loguniform(lo, hi)
        else: # list or constant
            if not isinstance(v, (list, tuple)):
                param_dists[k] = [v]
            else:
                param_dists[k] = v
    return sbatch_args, param_dists


def search(command,
           param_dists,
           num_samples,
           sbatch_args,
           source_bashrc=True,
           conda_env=None,
           hparams_save_path="./",
           max_concurrent_jobs=0,
           wait=False,
           random_seed=42):
    sampler = ParameterSampler(param_dists,
                               n_iter=num_samples,
                               random_state=random_seed)

    hparams = list(sampler)
    hparam_args = []
    for p in hparams:
        args_list = []
        for k, v in p.items():
            if isinstance(v, bool):
                if v:
                    args_list.append(f"--{k}")
            else:
                args_list.append(f"--{k} {v}")
        hparam_args.append(" ".join(args_list))

    hparams_array = "('" + "' '".join(hparam_args) + "')"
    script = ["#!/bin/bash\n"] + [f"#SBATCH {arg}\n" for arg in sbatch_args]
    script.append(f"#SBATCH --array=0-{len(hparams)-1}%{max_concurrent_jobs}\n")
    if wait:
       script.append("#SBATCH -W")
    if conda_env or source_bashrc:
        script.append("source ~/.bashrc\n")
    if conda_env:
        script.append(f"conda activate {conda_env}\n")

    script.append(f"HPARAMS={hparams_array}\n")
    script.append(f"{command} ${{HPARAMS[$SLURM_ARRAY_TASK_ID]}}\n")

    with tempfile.NamedTemporaryFile("w") as f:
        f.writelines(script)
        f.seek(0)
        out = subprocess.run(
            ["sbatch", os.path.join(tempfile.gettempdir(), f.name)],
            capture_output=True)

    job_id = out.stdout.decode().replace("Submitted batch job ",
                                         "").rstrip("\n")
    df = pd.DataFrame(hparams)
    df["version"] = [f"{job_id}_{str(i)}" for i in range(len(df))]
    df.to_csv(os.path.join(hparams_save_path,
                           f"hparams_{job_id}.csv"),
              index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--hparams_save_path", type=str, default="./")
    parser.add_argument("--num_samples", "-n", type=int, default=20)
    parser.add_argument("--max_concurrent_jobs", type=int, default=0)
    parser.add_argument("--source_bashrc", action="store_true")
    parser.add_argument("--conda_env", type=str)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    sbatch_args, param_dists = parse_config(args.config_file)
    search(args.command, param_dists, args.num_samples, sbatch_args,
           args.source_bashrc, args.conda_env, args.hparams_save_path,
           args.max_concurrent_jobs, args.wait, args.random_seed)

