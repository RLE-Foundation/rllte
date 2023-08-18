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


from termcolor import colored

from rllte.agent import PPO
from rllte.env.utils import make_rllte_env

if __name__ == "__main__":
    env = make_rllte_env(env_id="Acrobot-v1", num_envs=1, device="cpu")
    agent = PPO(env=env, device="cpu", tag="verification")
    try:
        agent.train(num_train_steps=1000)
        print(colored("Verification Passed!".upper(), "green", attrs=["bold"]))
        print(
            """
                 ___           ___       ___       ___           ___     
                /\  \         /\__\     /\__\     /\  \         /\  \    
               /::\  \       /:/  /    /:/  /     \:\  \       /::\  \   
              /:/\:\  \     /:/  /    /:/  /       \:\  \     /:/\:\  \  
             /::\-\:\  \   /:/  /    /:/  /        /::\  \   /::\-\:\  \ 
            /:/\:\ \:\__\ /:/__/    /:/__/        /:/\:\__\ /:/\:\ \:\__\\
            \/_|::\/:/  / \:\  \    \:\  \       /:/  \/__/ \:\-\:\ \/__/
               |:|::/  /   \:\  \    \:\  \     /:/  /       \:\ \:\__\  
               |:|\/__/     \:\  \    \:\  \    \/__/         \:\ \/__/  
               |:|  |        \:\__\    \:\__\                  \:\__\    
                \|__|         \/__/     \/__/                   \/__/    
            """
        )
    except RuntimeError:
        print(colored("Verification failed!".upper(), "red", attrs=["bold"]))
