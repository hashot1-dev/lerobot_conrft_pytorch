
docker run --runtime=nvidia --privileged -v \
/home/sr/.cache:/root/.cache -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=:1 \
-it --rm -v /home/sr/so101/lerobot:/workspace -w /workspace --ulimit memlock=-1 \
--ulimit stack=6700108864 --shm-size 7000000000 mk:lerobot1101