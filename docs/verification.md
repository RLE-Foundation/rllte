## Software
To ensure that **RLLTE** is installed correctly, we can verify the installation by running a single training script:
``` shell
python -m rllte.verification
```
If successful, you will see the following output:
``` sh
[08/03/2023 07:30:21 PM] - [INFO.] - Invoking RLLTE Engine...
[08/03/2023 07:30:21 PM] - [INFO.] - ================================================================================
[08/03/2023 07:30:21 PM] - [INFO.] - Tag               : verification
[08/03/2023 07:30:21 PM] - [INFO.] - Device            : CPU
[08/03/2023 07:30:21 PM] - [DEBUG] - Agent             : PPO
[08/03/2023 07:30:21 PM] - [DEBUG] - Encoder           : IdentityEncoder
[08/03/2023 07:30:21 PM] - [DEBUG] - Policy            : OnPolicySharedActorCritic
[08/03/2023 07:30:21 PM] - [DEBUG] - Storage           : VanillaRolloutStorage
[08/03/2023 07:30:21 PM] - [DEBUG] - Distribution      : Categorical
[08/03/2023 07:30:21 PM] - [DEBUG] - Augmentation      : False
[08/03/2023 07:30:21 PM] - [DEBUG] - Intrinsic Reward  : False
[08/03/2023 07:30:21 PM] - [DEBUG] - ================================================================================
[08/03/2023 07:30:22 PM] - [TRAIN] - S: 512         | E: 4           | L: 428         | R: -427.000    | FPS: 1457.513  | T: 0:00:00    
[08/03/2023 07:30:22 PM] - [TRAIN] - S: 640         | E: 5           | L: 428         | R: -427.000    | FPS: 1513.510  | T: 0:00:00    
[08/03/2023 07:30:22 PM] - [TRAIN] - S: 768         | E: 6           | L: 353         | R: -352.000    | FPS: 1551.423  | T: 0:00:00    
[08/03/2023 07:30:22 PM] - [TRAIN] - S: 896         | E: 7           | L: 353         | R: -352.000    | FPS: 1581.616  | T: 0:00:00    
[08/03/2023 07:30:22 PM] - [INFO.] - Training Accomplished!
[08/03/2023 07:30:22 PM] - [INFO.] - Model saved at: /export/yuanmingqi/code/rllte/logs/verification/2023-08-03-07-30-21/model
VERIFICATION PASSED!
```

## Hardware
Additionally, to check if your GPU driver and CUDA is enabled and accessible by PyTorch, run the following commands to return whether or not the CUDA driver is enabled:
``` python
import torch
torch.cuda.is_available()
```

For HUAWEI NPU:

``` python
import torch
import torch_npu
torch.npu.is_available()
```