#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _default_urdf_path() -> str:
    unitree_ros_dir = os.environ.get("UNITREE_ROS_DIR", "/home/ubuntu/unitree_ros")
    return str(Path(unitree_ros_dir) / "robots" / "g1_description" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf")


def _default_output_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "artifacts" / "usd" / "g1_inspire_dfq" / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd")


parser = argparse.ArgumentParser(description="Convert the G1 Inspire DFQ URDF into a USD file.")
parser.add_argument("--input", type=str, default=_default_urdf_path(), help="Path to the G1 Inspire DFQ URDF.")
parser.add_argument("--output", type=str, default=_default_output_path(), help="Path to write the generated USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def main():
    input_path = Path(args_cli.input).expanduser().resolve()
    output_path = Path(args_cli.output).expanduser().resolve()

    if not check_file_path(str(input_path)):
        raise FileNotFoundError(f"Invalid URDF path: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(input_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=False,
        collision_from_visuals=False,
        self_collision=False,
        replace_cylinders_with_capsules=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position",
        ),
    )

    print(f"Input URDF: {input_path}")
    print(f"Output USD: {output_path}")
    print_dict(cfg.to_dict(), nesting=0)

    converter = UrdfConverter(cfg)
    print(f"Generated USD file: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
