python -m lerobot.scripts.find_joint_limits \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM3 \
    --robot.id=follower1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader1