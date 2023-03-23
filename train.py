import numpy as np
aux_idx = np.arange(32 * 64)
np.random.shuffle(aux_idx)
print(aux_idx)

for i, start in enumerate(range(0, 32 * 64, 4)):
    end = start + 4
    print(end)