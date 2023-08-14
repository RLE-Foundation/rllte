# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import torch as th
from huggingface_hub import hf_hub_download
from torch import nn


class Procgen:
    """Trained models various RL algorithms on the full Procgen benchmark.
        Environment link: https://github.com/openai/procgen
        Number of environments: 16
        Number of training steps: 25,000,000
        Number of seeds: 10
        Added algorithms: [PPO]
    """

    def __init__(self) -> None:
        pass

    def load_models(
        self,
        agent: str,
        env_id: str,
        seed: int,
        device: str = "cpu",
    ) -> nn.Module:
        """Load the model from the hub.

        Args:
            agent (str): The agent to load.
            env_id (str): The environment id to load.
            seed (int): The seed to load.
            device (str): The device to load the model on.

        Returns:
            The loaded model.
        """
        model_file = f"{agent.lower()}_procgen_{env_id.lower()}_seed_{seed}.pth"
        subfolder = f"procgen/{agent}"
        file = hf_hub_download(repo_id="RLE-Foundation/rllte-hub", repo_type="model", filename=model_file, subfolder=subfolder)
        model = th.load(file, map_location=device)

        return model.eval()
