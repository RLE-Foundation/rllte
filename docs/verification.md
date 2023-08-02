## Software
To ensure that **RLLTE** is installed correctly, we can verify the installation by running a single training script:
``` shell
python -m rllte.verification
```
If successful, you will see the following output:
<div align=center>
<img src='../../assets/images/verification.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

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