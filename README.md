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

Run standalone deploy with GUI by default:

```bash
bash launch.sh deploy-standalone
```

Run direct ROS2 deploy with GUI by default:

```bash
bash launch.sh deploy-ros2
```

Run either deploy mode headless:

```bash
bash launch.sh deploy-standalone --headless
bash launch.sh deploy-ros2 --headless
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
- [policy.pt](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/artifacts/policies/default/policy.pt)

## Standalone Deploy Reproduction

学習 run `2026-03-14_13-45-52_default` を Isaac Sim standalone に再現 deploy する手順。

最短手順は次の 1 コマンド。

```bash
bash launch.sh deploy-standalone
```

- デフォルトは GUI あり
- `--headless` を付けると headless
- デフォルトでは repo 同梱の [policy.pt](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/artifacts/policies/default/policy.pt) を使う
- `--load-run` や `--checkpoint` を指定した場合は、必要なら自動 export
- 追加の Isaac Sim 引数はそのまま後ろに渡せる
  - 例: `bash launch.sh deploy-standalone --headless --num-steps 200 --lin-vel-x 0.6`

明示手順で追いたい場合は以下。

1. コンテナを起動する。

```bash
bash launch.sh up
```

2. 必要なら `docker/.env` を確認する。
   - 初回は `launch.sh` が [docker/.env.example](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/docker/.env.example) を自動コピーする。
   - DCV GUI を使う場合は `HOST_DISPLAY` と `HOST_XAUTHORITY_DIR` をホストに合わせる。

3. USD を再生成したい場合だけ変換を実行する。

```bash
docker exec -it isaaclab-g1-inspire-locomotion bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/convert_g1_inspire_usd.py --headless
'
```

4. 学習 checkpoint から TorchScript policy を export する。

```bash
docker exec -it isaaclab-g1-inspire-locomotion bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/export_policy_jit.py \
  --headless \
  --mode default \
  --load_run 2026-03-14_13-45-52_default \
  --checkpoint model_1499.pt
'
```

5. deploy 前の順序確認を headless で実行する。

```bash
docker exec -it isaaclab-g1-inspire-locomotion bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/check_deploy_headless.py \
  --headless \
  --mode default
'
```

6. standalone を headless で実行する。

```bash
docker exec -it isaaclab-g1-inspire-locomotion bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/isaac-sim/python.sh scripts/g1_standalone.py \
  --headless \
  --num-steps 200 \
  --policy-path logs/rsl_rl/g1_inspire_flat_default/2026-03-14_13-45-52_default/exported/policy.pt \
  --env-yaml logs/rsl_rl/g1_inspire_flat_default/2026-03-14_13-45-52_default/params/env.yaml \
  --usd-path artifacts/usd/g1_inspire_dfq/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd
'
```

7. GUI ありで確認したい場合は `--headless` を外す。
   - DCV 上の標準 GUI を使う場合も同じコマンドでよい。
   - 録画する場合は `--record-video --video-output outputs/g1_standalone.mp4` を追加する。

## Direct ROS2 Deploy

direct ROS2 bridge 方式を最短で試す場合も 1 コマンドでよい。

```bash
bash launch.sh deploy-ros2
```

- デフォルトは GUI あり
- `--headless` を付けると headless
- `isaaclab` と `ros2-policy` の compose service を自動で起動する
- ROS2 workspace の `g1_policy_controller` は自動 build する
- デフォルトでは repo 同梱の [policy.pt](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/artifacts/policies/default/policy.pt) を使う
- `--load-run` や `--checkpoint` を指定した場合は、必要なら自動 export する
- 追加の Isaac Sim bridge 引数もそのまま後ろに渡せる
  - 例: `bash launch.sh deploy-ros2 --headless --num-steps 200 --lin-vel-x 0.6`

実体としては以下を自動化している。

- `ros2-policy` コンテナで `ros2 run g1_policy_controller policy_node`
- `isaaclab` コンテナで `scripts/g1_ros2_bridge.py`

現在、安定している外部制御系はこの direct bridge 版で、Action Graph 版はまだ WIP。

## GUI 安定化メモ

standalone deploy 自体は headless では安定していたが、GUI ありでは短時間で転倒していた。
原因は policy や学習 run そのものではなく、GUI 実行時だけ simulation loop が別物になっていたことだった。

- 悪かった構成
  - `world.step(render=True)` や `simulation_app.update()` により、描画更新が physics/control loop に混ざっていた。
  - GUI では `isaaclab.python.kit` が動き、viewport や UI、render 系 extension の更新が loop に入っていた。
  - その結果、headless と GUI で state の時間発展が変わり、heading が早い段階で崩れて転倒していた。

- 解消した構成
  - physics は常に `world.step(render=False)` だけで進める。
  - GUI 更新は `simulation_app.update()` ではなく `world.render()` に分離する。
  - GUI 描画周期は physics 毎ではなく、少なくとも `decimation` 周期に落とす。
  - `/persistent/simulation/minFrameRate=1` を設定して、GUI 負荷による simulation clamping の影響を下げる。
  - DCV の X11 認証は xauth ファイルを直接 bind するのではなく、DCV のディレクトリごと mount する。

- 確認結果
  - headless では 5000 step 以上安定して歩行した。
  - GUI でも上記の修正後は、以前のような即時転倒が解消した。

要点は、`simulation が主、GUI はその結果を読むだけ` の構成に寄せること。
描画が simulation を駆動し始めると、headless では再現しない不安定性が出やすい。

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

## Portability Notes

clone 先でも動かしやすいように、docker compose 周りは次の前提に寄せている。

- `docker/.env` が無ければ [docker/.env.example](/home/ubuntu/isaaclab_ws/isaaclab-g1-inspire-locomotion/docker/.env.example) を自動コピーする
- cache はデフォルトで repo 相対の `.docker-cache/` を使う
- compose service は固定 `container_name` を使わない
- `COMPOSE_PROJECT_NAME` もデフォルト固定しない

つまり、別ディレクトリへ clone した場合でも、同じホスト上で元の checkout と衝突しにくい。
必要なら `docker/.env` に host 固有の `HOST_DISPLAY`, `HOST_XAUTHORITY_DIR`, `HOST_UNITREE_ROS_PATH` だけを上書きすればよい。
