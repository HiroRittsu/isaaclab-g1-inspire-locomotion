# isaaclab-g1-inspire-locomotion

External Isaac Lab tasks for G1 Inspire locomotion training.

Modes:
- `default`: simple flat gait bootstrap
- `advanced`: resume-friendly flat gait with xy/yaw and width penalty
- `loose_termination`: debug variant with looser termination thresholds
- `unitree_rewards`: flat variant that imports `unitree_rl_lab`-style reward terms

## Environment

- Isaac Lab runs through the docker compose service in [docker/docker-compose.yaml](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/docker/docker-compose.yaml)
- Main entrypoint is [launch.sh](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/launch.sh)
- Host-specific docker settings live in `docker/.env`; if missing, `launch.sh` copies [docker/.env.example](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/docker/.env.example) automatically.
- TensorBoard reads:
  - `/workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/logs`
- Default S3 root:
  - `s3://isaacsim-survey/isaaclab-g1-inspire-locomotion`

## Common Commands

Start container:

```bash
bash launch.sh up
```

Open shell in container:

```bash
bash launch.sh shell
```

Start TensorBoard:

```bash
bash launch.sh tensorboard 6006
```

Check TensorBoard:

```bash
bash launch.sh tensorboard-status
```

## Training

Run `default` training with video and automatic S3 upload:

```bash
bash launch.sh train --mode default --headless --num_envs 512 --video
```

Run `advanced` training from a checkpoint:

```bash
bash launch.sh train \
  --mode advanced \
  --headless \
  --num_envs 512 \
  --video \
  --resume \
  --load_run bootstrap_2026-03-12_16-13-20_default \
  --checkpoint model_1499.pt
```

Run without S3 upload:

```bash
bash launch.sh train --mode default --headless --num_envs 512 --video --no-s3
```

Notes:
- If `--video` is set and `--video_interval` is omitted, `launch.sh` computes an interval that targets about 20 videos over the whole training run.
- Train videos are uploaded to S3 automatically by default.
- Train videos are stored under:
  - `logs/rsl_rl/<experiment>/<run>/videos/train/`

## Play

Run playback with video and automatic S3 upload:

```bash
bash launch.sh play --mode default --video --load_run 2026-03-12_16-13-20_default --checkpoint model_1499.pt
```

Run playback without S3 upload:

```bash
bash launch.sh play --mode advanced --video --no-s3
```

Notes:
- Play videos are stored under:
  - `logs/rsl_rl/<experiment>/<run>/videos/play/`
- The latest play video is uploaded to S3 automatically by default.

## Artifacts

Tracked checkpoint artifact for the latest `default` policy:

- [model_1499.pt](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/artifacts/checkpoints/default/model_1499.pt)

## Deploy Reproducibility Notes

将来 Isaac Sim に standalone deploy することを考えると、学習時点で以下を固定または保存しておくと、
deploy 時の曖昧さをかなり減らせる。

- action joint の指定は、順序が重要な場合は regex だけでなく明示的な joint 名リストを使う。
- 各 run ごとに、学習環境で実際に解決されたロボット全 DoF 順を保存する。
- 各 run ごとに、action manager が実際に解決した action joint 順を保存する。
- 各 run ごとに、observation layout を保存する。
  - term の並び順
  - 各 term の次元
  - joint ベース observation block の joint 順
- raw checkpoint だけでなく、`policy.pt` も各 run で自動 export する。
- 各 run ごとに、deploy 用の manifest を小さくてもよいので保存する。
  - USD path または asset version
  - `dt`
  - `decimation`
  - action scale と offset mode
  - heading ベース yaw か direct yaw-rate かといった command mode
  - default joint pose
  - terrain と ground friction 設定
- もし deploy 側で direct yaw-rate command を使う予定なら、学習時から heading 由来 yaw ではなく、
  その command mode で学習しておく方が安全。
