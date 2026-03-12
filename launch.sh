#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.yaml"
SERVICE="isaaclab"
CONTAINER="isaaclab-g1-inspire-locomotion"
TB_PID_FILE="$ROOT_DIR/outputs/tensorboard.pid"
TB_LOG_FILE="$ROOT_DIR/outputs/tensorboard.log"

ensure_up() {
  docker compose --env-file "$ROOT_DIR/docker/.env" -f "$COMPOSE_FILE" up -d
}

ensure_installed() {
  docker compose --env-file "$ROOT_DIR/docker/.env" -f "$COMPOSE_FILE" exec -T "$SERVICE" bash -lc '
    cd /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion && \
    /isaac-sim/python.sh -m pip install -e . >/tmp/g1_inspire_pip.log 2>&1 || { cat /tmp/g1_inspire_pip.log; exit 1; }
  '
}

run_in_container() {
  docker compose --env-file "$ROOT_DIR/docker/.env" -f "$COMPOSE_FILE" exec -T "$SERVICE" bash -lc "$1"
}

upload_latest_video() {
  local s3_uri="$1"
  local latest
  latest=$(find "$ROOT_DIR/logs" -path "*/videos/play/*.mp4" -type f | sort | tail -n 1)
  test -n "$latest" && aws s3 cp "$latest" "$s3_uri"
}

sync_train_videos_once() {
  local s3_uri="$1"
  aws s3 sync "$ROOT_DIR/logs" "$s3_uri" --exclude "*" --include "*/videos/train/*.mp4" >/dev/null
}

start_train_video_sync_loop() {
  local s3_uri="$1"
  local interval_s="${2:-120}"
  (
    while true; do
      sync_train_videos_once "$s3_uri" || true
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
  echo "usage: $0 {up|down|shell|train|play|tensorboard|tensorboard-stop|tensorboard-status} ..." >&2
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
    video_sync_interval=120
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
        *)
          passthrough+=("$1")
          shift
          ;;
      esac
    done

    sync_pid=""
    if [[ -n "$s3_uri" ]]; then
      sync_pid=$(start_train_video_sync_loop "$s3_uri" "$video_sync_interval")
    fi

    set +e
    # shellcheck disable=SC2145
    run_in_container "/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/scripts/train.py ${passthrough[*]}"
    train_status=$?
    set -e

    if [[ -n "$sync_pid" ]]; then
      kill "$sync_pid" >/dev/null 2>&1 || true
      wait "$sync_pid" 2>/dev/null || true
      sync_train_videos_once "$s3_uri" || true
    fi
    exit "$train_status"
    ;;
  play)
    ensure_up
    ensure_installed
    s3_uri=""
    passthrough=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --s3-uri)
          s3_uri="$2"
          shift 2
          ;;
        *)
          passthrough+=("$1")
          shift
          ;;
      esac
    done
    # shellcheck disable=SC2145
    run_in_container "/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_ws/isaaclab-g1-inspire-locomotion/scripts/play.py ${passthrough[*]}"
    if [[ -n "$s3_uri" ]]; then
      upload_latest_video "$s3_uri"
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
    exit 1
    ;;
esac
