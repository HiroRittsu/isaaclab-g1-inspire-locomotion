#!/usr/bin/env python3
import argparse
import pathlib
import runpy
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "source" / "isaaclab_g1_inspire_locomotion"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import isaaclab_g1_inspire_locomotion  # noqa: F401

OFFICIAL_RSL_RL_DIR = pathlib.Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl")
if str(OFFICIAL_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_RSL_RL_DIR))

TASK_BY_MODE = {
    "default": "Isaac-G1-Inspire-Flat-Default-v0",
    "advanced": "Isaac-G1-Inspire-Flat-Advanced-v0",
    "loose_termination": "Isaac-G1-Inspire-Flat-LooseTermination-v0",
    "unitree_rewards": "Isaac-G1-Inspire-Flat-UnitreeRewards-v0",
}

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mode", choices=TASK_BY_MODE.keys(), default="default")
args, rest = parser.parse_known_args()

sys.argv = [
    "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
    "--task",
    TASK_BY_MODE[args.mode],
    *rest,
]
runpy.run_path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py", run_name="__main__")
