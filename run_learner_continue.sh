export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
# python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so101_learner.json
python -m lerobot.rl.learner --config_path=/workspace/outputs/train/2025-11-01/08-43-02_default/checkpoints/0015000/pretrained_model/train_config.json --resume=true --output_dir=/workspace/outputs/train/2025-11-01/08-43-02_default