#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yaml"
SERVICE="isaaclab"
CONTAINER="isaaclab-g1-inspire-locomotion"
TB_PID_FILE="$ROOT_DIR/outputs/tensorboard.pid"
TB_LOG_FILE="$ROOT_DIR/outputs/tensorboard.log"
DEFAULT_S3_ROOT="s3://isaacsim-survey/isaaclab-g1-inspire-locomotion"
ENV_FILE="$ROOT_DIR/docker/.env"
ENV_EXAMPLE_FILE="$ROOT_DIR/docker/.env.example"

usage() {
  cat <<'EOF'
usage: launch.sh {up|down|shell|train|play|tensorboard|tensorboard-stop|tensorboard-status} ...

commands:
  up
    Start the Isaac Lab docker compose service.

  down
    Stop the Isaac Lab docker compose service.

  shell
    Open a shell inside the Isaac Lab container.

  train [args...]
    Run training inside the container.
    Common args:
      --mode {default|advanced|loose_termination|unitree_rewards}
      --headless
      --num_envs N
      --video
      --video_length N
      --video_interval N
      --max_iterations N
      --resume --load_run RUN --checkpoint FILE
      --no-s3
      --s3-uri s3://...
      --video-sync-interval SEC

    Notes:
      - If --video is enabled and --video_interval is omitted, the script sets an interval
        that targets about 20 videos over the whole training run.
      - Train videos are uploaded to S3 automatically by default.

  play [args...]
    Run policy playback inside the container.
    Common args:
      --mode {default|advanced|loose_termination|unitree_rewards}
      --video
      --checkpoint FILE
      --load_run RUN
      --no-s3
      --s3-uri s3://...

    Notes:
      - The latest play video is uploaded to S3 automatically by default.

  tensorboard [port]
    Start TensorBoard on the host network. Default port is 6006.

  tensorboard-stop
    Stop TensorBoard.

  tensorboard-status
    Show TensorBoard status.
EOF
}

ensure_up() {
  ensure_env_file
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
}

ensure_installed() {
  ensure_env_file
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec -T "$SERVICE" bash -lc '
    cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion && \
    /isaac-sim/python.sh -m pip install -e . >/tmp/g1_inspire_pip.log 2>&1 || { cat /tmp/g1_inspire_pip.log; exit 1; }
  '
}

run_in_container() {
  ensure_env_file
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec -T "$SERVICE" bash -lc "$1"
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return 0
  fi

  if [[ ! -f "$ENV_EXAMPLE_FILE" ]]; then
    echo "missing env template: $ENV_EXAMPLE_FILE" >&2
    exit 1
  fi

  cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
  echo "created $ENV_FILE from $ENV_EXAMPLE_FILE; adjust host-specific values if needed." >&2
}

parse_mode_from_args() {
  local default_mode="default"
  local args=("$@")
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--mode" && $((i + 1)) -lt ${#args[@]} ]]; then
      echo "${args[$((i + 1))]}"
      return 0
    fi
    ((i += 1))
  done
  echo "$default_mode"
}

has_arg() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

parse_num_envs_from_args() {
  local default_num_envs="512"
  local args=("$@")
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--num_envs" && $((i + 1)) -lt ${#args[@]} ]]; then
      echo "${args[$((i + 1))]}"
      return 0
    fi
    ((i += 1))
  done
  echo "$default_num_envs"
}

parse_max_iterations_from_args() {
  local args=("$@")
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--max_iterations" && $((i + 1)) -lt ${#args[@]} ]]; then
      echo "${args[$((i + 1))]}"
      return 0
    fi
    ((i += 1))
  done
  echo ""
}

default_max_iterations_for_mode() {
  case "$1" in
    default|loose_termination|unitree_rewards) echo "1500" ;;
    advanced) echo "2000" ;;
    *)
      echo "unknown mode: $1" >&2
      return 1
      ;;
  esac
}

default_video_interval_for_training() {
  local mode="$1"
  local max_iterations="$2"
  local num_steps_per_env=24
  local target_videos=20
  if [[ -z "$max_iterations" ]]; then
    max_iterations="$(default_max_iterations_for_mode "$mode")"
  fi
  if [[ "$target_videos" -le 1 ]]; then
    echo "$num_steps_per_env"
    return 0
  fi
  local total_env_steps=$((max_iterations * num_steps_per_env))
  local interval=$(((total_env_steps + target_videos - 2) / (target_videos - 1)))
  if [[ "$interval" -lt 1 ]]; then
    interval=1
  fi
  echo "$interval"
}

experiment_name_for_mode() {
  case "$1" in
    default) echo "g1_inspire_flat_default" ;;
    advanced) echo "g1_inspire_flat_advanced" ;;
    loose_termination) echo "g1_inspire_flat_loose_termination" ;;
    unitree_rewards) echo "g1_inspire_flat_unitree_rewards" ;;
    *)
      echo "unknown mode: $1" >&2
      return 1
      ;;
  esac
}

default_train_s3_uri() {
  local mode="$1"
  echo "${DEFAULT_S3_ROOT}/train/${mode}/"
}

default_play_s3_uri() {
  local mode="$1"
  echo "${DEFAULT_S3_ROOT}/play/${mode}/"
}

upload_latest_video() {
  local source_dir="$1"
  local s3_uri="$2"
  local latest
  latest=$(find "$source_dir" -path "*/videos/play/*.mp4" -type f | sort | tail -n 1)
  if [[ -n "$latest" ]]; then
    local run_dir
    run_dir="$(basename "$(dirname "$(dirname "$latest")")")"
    aws s3 cp "$latest" "${s3_uri}${run_dir}/$(basename "$latest")"
  fi
}

sync_train_videos_once() {
  local source_dir="$1"
  local s3_uri="$2"
  aws s3 sync "$source_dir" "$s3_uri" --exclude "*" --include "*/videos/train/*.mp4" >/dev/null
}

start_train_video_sync_loop() {
  local source_dir="$1"
  local s3_uri="$2"
  local interval_s="${3:-300}"
  (
    while true; do
      sync_train_videos_once "$source_dir" "$s3_uri" || true
      sleep "$interval_s"
    done
  ) &
  echo $!
}

is_port_listening() {
  local port="$1"
  ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .
}

tensorboard_start() {
  local port="${1:-6006}"
  ensure_up
  mkdir -p "$ROOT_DIR/outputs"

  if is_port_listening "$port"; then
    echo "tensorboard already listening on port $port" >&2
    return 0
  fi

  : > "$TB_LOG_FILE"
  docker exec -d "$CONTAINER" /isaac-sim/python.sh -m tensorboard.main \
    --logdir /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/logs \
    --host 0.0.0.0 \
    --port "${port}"
  sleep 5
  if ! is_port_listening "$port"; then
    echo "failed to start tensorboard" >&2
    return 1
  fi
  echo "$port" > "$TB_PID_FILE"
  echo "tensorboard started on port $port"
}

tensorboard_stop() {
  set +e
  run_in_container "pkill -f '/isaac-sim/kit/python/bin/python3 -m tensorboard.main --logdir /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/logs' >/dev/null 2>&1 || true"
  set -e
  rm -f "$TB_PID_FILE"
  echo "tensorboard stopped"
}

tensorboard_status() {
  if is_port_listening 6006; then
    echo "6006" > "$TB_PID_FILE"
    echo "tensorboard running on port 6006"
  else
    echo "tensorboard is not running"
    rm -f "$TB_PID_FILE"
  fi
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage >&2
  exit 1
fi
shift || true

case "$cmd" in
  up)
    ensure_up
    ;;
  down)
    docker compose --env-file "$ROOT_DIR/docker/.env" -f "$COMPOSE_FILE" down
    ;;
  shell)
    ensure_up
    ensure_installed
    docker compose --env-file "$ROOT_DIR/docker/.env" -f "$COMPOSE_FILE" exec "$SERVICE" bash
    ;;
  train)
    ensure_up
    ensure_installed
    s3_uri=""
    enable_s3=1
    video_sync_interval=300
    passthrough=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --s3-uri)
          s3_uri="$2"
          shift 2
          ;;
        --video-sync-interval)
          video_sync_interval="$2"
          shift 2
          ;;
        --no-s3)
          enable_s3=0
          shift
          ;;
        *)
          passthrough+=("$1")
          shift
          ;;
      esac
    done

    mode="$(parse_mode_from_args "${passthrough[@]}")"
    num_envs="$(parse_num_envs_from_args "${passthrough[@]}")"
    max_iterations="$(parse_max_iterations_from_args "${passthrough[@]}")"
    experiment_name="$(experiment_name_for_mode "$mode")"
    source_dir="$ROOT_DIR/logs/rsl_rl/$experiment_name"
    if [[ $enable_s3 -eq 1 && -z "$s3_uri" ]]; then
      s3_uri="$(default_train_s3_uri "$mode")"
    fi
    if has_arg "--video" "${passthrough[@]}" && ! has_arg "--video_interval" "${passthrough[@]}"; then
      passthrough+=("--video_interval" "$(default_video_interval_for_training "$mode" "$max_iterations")")
    fi

    sync_pid=""
    if [[ -n "$s3_uri" ]]; then
      sync_pid=$(start_train_video_sync_loop "$source_dir" "$s3_uri" "$video_sync_interval")
    fi

    set +e
    # shellcheck disable=SC2145
    run_in_container "/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/scripts/train.py ${passthrough[*]}"
    train_status=$?
    set -e

    if [[ -n "$sync_pid" ]]; then
      kill "$sync_pid" >/dev/null 2>&1 || true
      wait "$sync_pid" 2>/dev/null || true
      sync_train_videos_once "$source_dir" "$s3_uri" || true
    fi
    exit "$train_status"
    ;;
  play)
    ensure_up
    ensure_installed
    s3_uri=""
    enable_s3=1
    passthrough=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --s3-uri)
          s3_uri="$2"
          shift 2
          ;;
        --no-s3)
          enable_s3=0
          shift
          ;;
        *)
          passthrough+=("$1")
          shift
          ;;
      esac
    done
    mode="$(parse_mode_from_args "${passthrough[@]}")"
    experiment_name="$(experiment_name_for_mode "$mode")"
    source_dir="$ROOT_DIR/logs/rsl_rl/$experiment_name"
    if [[ $enable_s3 -eq 1 && -z "$s3_uri" ]]; then
      s3_uri="$(default_play_s3_uri "$mode")"
    fi
    # shellcheck disable=SC2145
    run_in_container "/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/scripts/play.py ${passthrough[*]}"
    if [[ -n "$s3_uri" ]]; then
      upload_latest_video "$source_dir" "$s3_uri"
    fi
    ;;
  tensorboard)
    tensorboard_start "${1:-6006}"
    ;;
  tensorboard-stop)
    tensorboard_stop
    ;;
  tensorboard-status)
    tensorboard_status
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
