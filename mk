docker run   --runtime=nvidia   -v ~/.cache:/root/.cache -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1  -it --rm      -v "$PWD":/workspace     -w /workspace  --ulimit memlock=-1 --ulimit stack=6700108864  --shm-size 7000000000 mk:lerobot


docker run   --runtime=nvidia  \
--device=/dev/video4 \
--device=/dev/video2 \
--device=/dev/ttyACM0 \
--device=/dev/ttyACM1 \
--device=/dev/ttyACM2 \
 -v ~/.cache:/root/.cache \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -e DISPLAY=:1  -it --rm   \
    -v "$PWD":/workspace \
        -w /workspace  --ulimit memlock=-1 --ulimit stack=6700108864   --shm-size 7000000000 mk:lerobot1
