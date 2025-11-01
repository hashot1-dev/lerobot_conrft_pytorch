#!/bin/bash

# List of device strings
device_list=(
  "/dev/ttyACM0"
  "/dev/ttyACM1"
  "/dev/ttyACM2"
  "/dev/ttyACM3"
  "/dev/ttyACM4"
)

# Loop through each device in the list
for stringx in "${device_list[@]}"; do
  # Check for "follower" identifier (5AAF219986)
  result_follower=$(udevadm info "$stringx" | grep "5AAF219986")

  # Check for "leader" identifier (5AAF218179)
  result_leader=$(udevadm info "$stringx" | grep "5AAF218179")

  # Determine if it's a follower or leader based on the result
  if [[ -n "$result_follower" ]]; then
    echo "$stringx : follower!!!!"
    echo $stringx > devfollower
  elif [[ -n "$result_leader" ]]; then
    echo "$stringx : leader!!!!"

  fi


done
device_list=(
  "/dev/video1"
  "/dev/video2"
  "/dev/video3"
  "/dev/video4"
  "/dev/video5"
  "/dev/video6"
  "/dev/video7"
  "/dev/video8"
  "/dev/video9"
)

# Loop through each device in the list
for stringx in "${device_list[@]}"; do
  # Check for "follower" identifier (5AAF219986)
  result_top1=$(udevadm info "$stringx" | grep "243323065486")
  result_top2=$(udevadm info "$stringx" | grep "Camera_455_243323065486-video-index0")

  # Check for "leader" identifier (5AAF218179)
  result_right1=$(udevadm info "$stringx" | grep "44434000_P030C01_SN0002" )
  result_right2=$(udevadm info "$stringx" | grep ":capture:")

  # Determine if it's a follower or leader based on the result
  if [[ -n "$result_top1" ]] && [[ -n "$result_top2" ]] ; then
    echo "$stringx : top!!!!"
    echo $stringx > devtop
  elif [[ -n "$result_right1" ]] && [[ -n "$result_right2" ]]; then
    echo "$stringx : right!!!!"
    echo $stringx > devright

  fi
done