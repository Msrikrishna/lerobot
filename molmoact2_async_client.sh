#!/usr/bin/env bash
# =============================================================================
# molmoact2_async_client.sh
# -----------------------------------------------------------------------------
# Run the LeRobot async inference RobotClient on your robot host (Mac + SO-101).
# The policy server must already be running on the H100.
#
# Connectivity: the Hyperstack H100 has no open ports, so reach the server via
# an SSH tunnel from this Mac, in a separate terminal:
#
#     brev shell molmoact2-h100            # or: ssh <brev-ssh-alias>
#     # then, to tunnel:
#     ssh -N -L 8080:localhost:8080 <brev-ssh-alias>
#
# With the tunnel up, the server is reachable at 127.0.0.1:8080.
#
# IMPORTANT:
#   - PRETRAINED is a path *on the H100* (where the HF->LeRobot conversion saved
#     the checkpoint), NOT a path on this Mac.
#   - CAMERAS keys must exactly match the image_keys used during conversion.
#     You have one camera, so we use a single key here ("top"). Find its index
#     with:  lerobot-find-cameras
#
# Usage:
#   PRETRAINED=/home/ubuntu/molmoact2_so101_lerobot bash molmoact2_async_client.sh
# =============================================================================

set -euo pipefail

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"          # via SSH tunnel
ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5B7B0166391}"     # SO-101 follower
ROBOT_ID="${ROBOT_ID:-so101}"
TASK="${TASK:-pick up the book and place it}"
# Path to the converted LeRobot checkpoint ON THE H100 (2-camera conversion):
PRETRAINED="${PRETRAINED:-/home/shadeform/molmoact2_so101_lerobot_2cam}"
# Two cameras. Keys (wrist, side) MUST match the conversion image_keys
# (observation.images.wrist, observation.images.side).
#   index 0 = wrist / hand / ego cam
#   index 2 = side / third-person cam
WRIST_INDEX="${WRIST_INDEX:-0}"
SIDE_INDEX="${SIDE_INDEX:-2}"
# NOTE: build CAMERAS WITHOUT embedding the literal { } braces inside a
# ${VAR:-default} expansion — bash drops a brace there and corrupts the dict.
if [ -z "${CAMERAS:-}" ]; then
  CAMERAS="{ wrist: {type: opencv, index_or_path: ${WRIST_INDEX}, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: ${SIDE_INDEX}, width: 640, height: 480, fps: 30} }"
fi

python -m lerobot.async_inference.robot_client \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS" \
  --task="$TASK" \
  --policy_type=molmoact2 \
  --pretrained_name_or_path="$PRETRAINED" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=30 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
