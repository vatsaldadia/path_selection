import argparse
import json
import subprocess
from collections import defaultdict
import os
import torch

from stable_baselines3.common.env_util import make_vec_env
from sympy.physics.optics import jones_2_stokes

from continual_dqn import ContinualDQN
from drift_detector import DriftDetector
from generator import Generator
from testing import test_all_durations


def main():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--duration", type=str, default="month", help="week or month", choices=["week", "month"])
    parser.add_argument("--offset", type=int, default=0, help="Offset value")
    parser.add_argument("--num", type=int, default=1, help="Number of samples")
    parser.add_argument("--mode", type=str, default="continual", choices=["continual", "transfer", "online", "offline"])
    parser.add_argument("--test_all_durations", type=bool, default=True, help="Test on all durations")
    args = parser.parse_args()

    job_id = f"{args.mode}_{args.duration}_"
    for i in range(args.num):
        job_id += str(args.offset + i)
        if i != args.num - 1:
            job_id += ","
        else:
            job_id += "#"
    script_run = subprocess.run(["./delete.sh", job_id], capture_output=True, text=True)
    print(script_run.stdout, end="")

    generator = Generator()
    if args.mode == "transfer":
        traffics = [f"{args.duration}_{args.offset + (args.num - 1)}.csv"]
    elif args.mode == "continual" or args.mode == "online" or args.mode == "offline":
        traffics = [f"{args.duration}_{args.offset + i}.csv" for i in range(args.num)]
    else:
        raise Exception(f"Don't know how to handle mode {args.mode} while loading traffic")
    for traffic_file in traffics:
        generator.load_traffic(f"{traffic_file}")

    if args.mode == "online":
        timesteps = generator.get_total_traffic_count()
        threshold = 24 * args.num
    elif args.mode == "continual" or args.mode == "transfer":
        timesteps = 200000
        threshold = 24
    elif args.mode == "offline":
        timesteps = 200000 * args.num
        threshold = 24 * args.num
    else:
        raise Exception(f"Don't know how to handle mode {args.mode} while setting timesteps and threshold")

    drift_detector = DriftDetector(threshold=threshold)

    print(f"Creating job {job_id} with {threshold=} and {timesteps=}")

    continual_dqn = ContinualDQN(
        generator=generator,
        detector=drift_detector,
        job_id=job_id,
        mode=args.mode,
        timesteps=timesteps,
        epochs=50000,
        ewc_lambda=0.4,
        dynamic_end_times=False
    )
    continual_dqn.train()

    print("\n----------------- TESTING ------------------")
    continual_dqn.test(24 * args.num)

    if args.test_all_durations:
        print("\n----------------- CL TESTING ------------------")
        for i in range(0, 12):
            print(f"Testing job {job_id} with traffic {args.duration}_{i}")
            test_all_durations(job_id, f"{args.duration}_{i}")

if __name__ == "__main__":
    main()