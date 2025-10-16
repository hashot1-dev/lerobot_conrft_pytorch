wandb
7f00f9e062418963f09b09b5a137879fb8a8380b

docker run   --runtime=nvidia   -it --rm      -v "$PWD":/workspace     -w /workspace     nvcr.io/nvidia/pytorch:25.08-py3



docker run   --runtime=nvidia   -it --rm      -v "$PWD":/workspace     -w /workspace     mk:test1

Cal-ConRFT (offline)

rm outputs/ -rf
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
python src/lerobot/scripts/rl/learner.py --config json/train_conrft_offline.json

HIL-ConRFT (online)

python src/lerobot/scripts/rl/learner.py --config json/train_conrft_online_learner.json
python src/lerobot/scripts/rl/actor.py --config json/train_conrft_online_actor.json
You can find config files in this branch lilkm/configs here : https://github.com/s1lent4gnt/lerobot/tree/lilkm/configs/json



ERROR: Could not find a version that satisfies the requirement opencv-python==4 (from versions: 3.4.0.14, 3.4.10.37, 3.4.11.41, 3.4.11.43, 3.4.11.45, 3.4.13.47, 3.4.15.55, 3.4.16.57, 3.4.16.59, 3.4.17.63, 3.4.18.65, 4.3.0.38, 4.4.0.40, 4.4.0.46, 4.5.1.48, 4.5.3.56, 4.5.4.60, 4.5.5.64, 4.6.0.66, 4.7.0.72, 4.8.0.74, 4.8.0.76, 4.8.1.78, 4.9.0.80, 4.10.0.82, 4.10.0.84, 4.11.0.86, 4.12.0.88)