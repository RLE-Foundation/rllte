## Software
To ensure that PyTorch was installed correctly, we can verify the installation by running a single training script:
``` shell
python -m hsuanwu.verification
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