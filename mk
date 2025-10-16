wandb的 key  
7f00f9e062418963f09b09b5a137879fb8a8380b

1）docker启动命令：
cd  /home/sr/project/lerobot-lilkm-port-conrft/src
docker run   --runtime=nvidia   -it --rm      -v "$PWD":/workspace     -w /workspace     mk:test2

2)  Cal-ConRFT (offline)  执行 

rm outputs/ -rf
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
python src/lerobot/scripts/rl/learner.py --config json/train_conrft_offline.json

3） HIL-ConRFT (online)

python src/lerobot/scripts/rl/learner.py --config json/train_conrft_online_learner.json
python src/lerobot/scripts/rl/actor.py --config json/train_conrft_online_actor.json

You can find config files in this branch lilkm/configs here : https://github.com/s1lent4gnt/lerobot/tree/lilkm/configs/json

4) 注意事项  
torch 的版本 是定的，不要重新装 