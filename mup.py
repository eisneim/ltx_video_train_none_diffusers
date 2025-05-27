import subprocess, os, time, random
import torch
import matplotlib.pyplot as plt
import numpy as np

"""
run multi iterations to find best learning rate and batch_size
"""
def main():
    batch_sizes = [ 8 ]
    lrs = [x * 5e-6 for x in range(1, 16)]

    for bb in batch_sizes:
        for idx, lr in enumerate(lrs):
            print(f"[{idx} / {len(lrs)}] running for: batch_size {bb} lr: {lr}")
            base = "accelerate launch --config_file ./configs/uncompiled_2.yaml ltx_train.py "
            command = base + f"--batch_size {bb} --lr {lr}"
            result = subprocess.run([ command ], shell=True, capture_output=False, text=True)


def plot():
    pass


if __name__ == "__main__":
    main()

