from huggingface_hub import HfApi, login

login()

api = HfApi()
api.upload_file(
    path_or_fileobj="F:\\rllte-hub\\ppo_procgen\\procgen\\ppo\\ppo_procgen_bigfish_seed_1.pth",
    path_in_repo="procgen/ppo/ppo_procgen_bigfish_seed_1.pth",
    repo_id="RLE-Foundation/rllte-hub",
    repo_type="model",
)
