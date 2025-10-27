python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=mengke/so101_test_6 \
  --control.tags='["so101","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.display_data=true \
  --control.push_to_hub=false



  python lerobot/scripts/control_robot.py   --robot.type=so101   --control.type=record   --control.fps=30   --control.single_task="Grasp a lego block and put it in the bin."   --control.repo_id=mengke/so101_test_7   --control.tags='["so101","tutorial"]'   --control.warmup_time_s=5   --control.episode_time_s=30   --control.reset_time_s=1   --control.num_episodes=10   --control.display_data=true   --control.push_to_hub=false  
