# isaaclab-g1-inspire-locomotion

G1 Inspire の locomotion 学習と deploy を扱う Isaac Lab 用の外部タスク集です。

## 1. 概要

このリポジトリでは、主に次の 4 つのモードを使います。

- `default`: 平地での基本歩容を立ち上げる最小構成
- `advanced`: `xy/yaw` command と追加 reward を含む拡張構成
- `loose_termination`: termination 条件を緩めたデバッグ用構成
- `unitree_rewards`: `unitree_rl_lab` 風の reward を取り込んだ構成

## 2. 環境構成

- Isaac Lab 本体の docker compose は `docker/docker-compose.yaml`
- 外部 ROS2 policy 用の compose は `docker/docker-compose.ros2.yaml`
- 主な起動入口は `launch.sh`
- `docker/.env` が無ければ `launch.sh` が `docker/.env.example` を自動コピー
- TensorBoard の対象ログは `/workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/logs`
- 既定の S3 ルートは `s3://isaacsim-survey/isaaclab-g1-inspire-locomotion`

## 3. 基本コマンド

コンテナ起動:

```bash
bash launch.sh up
```

Isaac Lab コンテナに入る:

```bash
bash launch.sh shell
```

standalone deploy を GUI ありで起動:

```bash
bash launch.sh deploy-standalone
```

direct ROS2 deploy を GUI ありで起動:

```bash
bash launch.sh deploy-ros2
```

headless で起動:

```bash
bash launch.sh deploy-standalone --headless
bash launch.sh deploy-ros2 --headless
```

TensorBoard 起動:

```bash
bash launch.sh tensorboard 6006
```

TensorBoard 状態確認:

```bash
bash launch.sh tensorboard-status
```

## 4. 学習

`default` モードを video 付きで学習:

```bash
bash launch.sh train --mode default --headless --num_envs 512 --video
```

checkpoint から `advanced` を再開:

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

S3 upload を無効化:

```bash
bash launch.sh train --mode default --headless --num_envs 512 --video --no-s3
```

補足:

- `--video` を付けて `--video_interval` を省略すると、`launch.sh` が全体でおよそ 20 本になるよう自動計算する
- train video は既定で自動 S3 upload される
- train video の保存先は `logs/rsl_rl/<experiment>/<run>/videos/train/`

## 5. 再生

video 付きで playback:

```bash
bash launch.sh play --mode default --video --load_run 2026-03-12_16-13-20_default --checkpoint model_1499.pt
```

S3 upload なしで playback:

```bash
bash launch.sh play --mode advanced --video --no-s3
```

補足:

- play video の保存先は `logs/rsl_rl/<experiment>/<run>/videos/play/`
- 最新の play video は既定で自動 S3 upload される

## 6. 同梱 artifact

既定の deploy では、次の artifact をそのまま使えます。

- `artifacts/checkpoints/default/model_1499.pt`
- `artifacts/policies/default/policy.pt`

## 7. Standalone Deploy

学習 run `2026-03-14_13-45-52_default` を Isaac Sim standalone へ再現 deploy する最短手順です。

最短コマンド:

```bash
bash launch.sh deploy-standalone
```

補足:

- デフォルトは GUI あり
- `--headless` を付けると headless
- 既定では repo 同梱の `artifacts/policies/default/policy.pt` を使う
- `--load-run` や `--checkpoint` を付けた場合は、必要なら自動 export する
- 追加の Isaac Sim 引数はそのまま後ろへ渡せる
- 例:

```bash
bash launch.sh deploy-standalone --headless --num-steps 200 --lin-vel-x 0.6
```

明示手順で追う場合:

1. コンテナを起動する

```bash
bash launch.sh up
```

2. 必要なら `docker/.env` を確認する

- 初回は `launch.sh` が `docker/.env.example` を自動コピーする
- DCV GUI を使う場合は `HOST_DISPLAY` と `HOST_XAUTHORITY_DIR` をホストに合わせる

3. 必要な場合のみ USD を再生成する

```bash
docker compose -f docker/docker-compose.yaml exec isaaclab bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/convert_g1_inspire_usd.py --headless
'
```

4. 必要な場合のみ TorchScript policy を export する

```bash
docker compose -f docker/docker-compose.yaml exec isaaclab bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/export_policy_jit.py \
  --headless \
  --mode default \
  --load_run 2026-03-14_13-45-52_default \
  --checkpoint model_1499.pt
'
```

5. deploy 前の順序確認を headless で実行する

```bash
docker compose -f docker/docker-compose.yaml exec isaaclab bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/workspace/isaaclab/isaaclab.sh -p scripts/check_deploy_headless.py \
  --headless \
  --mode default
'
```

6. standalone を直接実行する

```bash
docker compose -f docker/docker-compose.yaml exec isaaclab bash -lc '
cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion &&
/isaac-sim/python.sh scripts/g1_standalone.py \
  --headless \
  --num-steps 200 \
  --policy-path artifacts/policies/default/policy.pt \
  --env-yaml logs/rsl_rl/g1_inspire_flat_default/2026-03-14_13-45-52_default/params/env.yaml \
  --usd-path artifacts/usd/g1_inspire_dfq/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd
'
```

7. GUI ありで確認したい場合は `--headless` を外す

- DCV 上の標準 GUI でも同じコマンドでよい
- 録画したい場合は `--record-video --video-path outputs/g1_standalone.mp4` を追加する

## 8. Direct ROS2 Deploy

安定している外部制御系は direct ROS2 bridge 版です。

最短コマンド:

```bash
bash launch.sh deploy-ros2
```

補足:

- デフォルトは GUI あり
- `--headless` を付けると headless
- `isaaclab` と `ros2-policy` の compose service を自動起動する
- ROS2 workspace の `g1_policy_controller` を自動 build する
- 既定では repo 同梱の `artifacts/policies/default/policy.pt` を使う
- `--load-run` や `--checkpoint` を付けた場合は、必要なら自動 export する
- 追加の Isaac Sim bridge 引数はそのまま後ろへ渡せる
- 例:

```bash
bash launch.sh deploy-ros2 --headless --num-steps 200 --lin-vel-x 0.6
```

内部で自動実行しているもの:

- `ros2-policy` コンテナで `ros2 run g1_policy_controller policy_node`
- `isaaclab` コンテナで `scripts/g1_ros2_bridge.py`

補足:

- Action Graph 版はまだ WIP
- すぐに外部 deploy を試すなら direct bridge 版を使う

## 9. GUI 安定化メモ

standalone deploy は headless では安定していた一方、GUI ありでは初期に転倒していました。
原因は policy や学習 run ではなく、GUI 実行時だけ simulation loop の責務分離が崩れていたことです。

悪かった構成:

- `world.step(render=True)` や `simulation_app.update()` により、描画更新が physics/control loop に混ざっていた
- GUI では `isaaclab.python.kit` が動き、viewport や UI、render 系 extension の更新が loop に入っていた
- その結果、headless と GUI で状態遷移が変わり、heading が早期に崩れていた

解消した構成:

- physics は常に `world.step(render=False)` だけで進める
- GUI 更新は `simulation_app.update()` ではなく `world.render()` に分離する
- GUI 描画周期は physics 毎ではなく、少なくとも `decimation` 周期へ落とす
- `/persistent/simulation/minFrameRate=1` を設定して simulation clamping の影響を下げる
- DCV の X11 認証は xauth ファイル直指定ではなく、DCV ディレクトリごとの mount にする

確認結果:

- headless では 5000 step 以上安定
- GUI でも上記修正後は即時転倒が解消

要点:

- `simulation が主、GUI は結果を読むだけ` に寄せる
- 描画が simulation を駆動し始めると、headless では再現しない不安定性が出やすい

## 10. 再現性向上メモ

将来の Isaac Sim deploy を確実にするため、学習時点では次を固定または保存しておくのがよいです。

- action joint 指定は regex だけでなく明示 joint 名リストで残す
- 学習環境で実際に解決された全 DoF 順を run ごとに保存する
- action manager が実際に解決した action joint 順を run ごとに保存する
- observation layout を run ごとに保存する
- raw checkpoint だけでなく `policy.pt` も自動 export する
- deploy 用 manifest を保存する

manifest へ入れておくとよい項目:

- USD path または asset version
- `dt`
- `decimation`
- action scale と offset mode
- heading ベース yaw か direct yaw-rate かという command mode
- default joint pose
- terrain と ground friction

補足:

- deploy 側で direct yaw-rate command を使う予定なら、学習時からその command mode に揃える方が安全

## 11. ポータビリティ

別ディレクトリへ clone した場合でも動かしやすいよう、docker compose 周りは次の方針にしています。

- `docker/.env` が無ければ `docker/.env.example` を自動コピーする
- cache の既定値は repo 相対の `.docker-cache/`
- compose service は固定 `container_name` を使わない
- `COMPOSE_PROJECT_NAME` も既定では固定しない

このため、同じホスト上に別 checkout があっても衝突しにくい構成です。
必要なら `docker/.env` で次だけ host 固有値へ上書きします。

- `HOST_DISPLAY`
- `HOST_XAUTHORITY_DIR`
- `HOST_UNITREE_ROS_PATH`
