# ros2_ws

G1 policy controller workspace for external ROS 2 execution.

## Build

```bash
source /opt/ros/humble/setup.bash
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/ros2_ws
colcon build --symlink-install
```

## Run policy node

```bash
source /opt/ros/humble/setup.bash
source /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/ros2_ws/install/setup.bash
ros2 run g1_policy_controller policy_node \
  --policy-path /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/logs/rsl_rl/g1_inspire_flat_default/2026-03-14_13-45-52_default/exported/policy.pt
```
