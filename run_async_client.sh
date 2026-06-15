#!/usr/bin/env bash
# Async-inference robot client: streams SO-101 obs to the H100 policy server.
# Make sure the SSH tunnel (ssh -N -L 8080:localhost:8080 my-pi05) is running first.
set -euo pipefail

python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0166391 \
  --robot.id=so101 \
  --robot.cameras="{ hand_cam: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 30}}" \
  --task="pick up the book and place it vertical" \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=srik410/pi05_book_parallel_grasp \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average
