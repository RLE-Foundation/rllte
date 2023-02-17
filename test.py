from pathlib import Path

replay_dir = Path.cwd() / './logs/buffer'

# eps_file = sorted(replay_dir.glob('*.npz'), reverse=True)
# for file in eps_file:
#     if file.stem == '1':
#         file.unlink(missing_ok=True)
#     print(file)

# eps_file = sorted(replay_dir.glob('*.npz'), reverse=True)
# print(eps_file)

# for file in eps_file:
#     print(file)

import numpy as np
import time
import io
from collections import defaultdict

buffer = []

for i in range(1):
    # print(i)
    buffer.append(np.load(replay_dir / f'{i}.npz'))
    # episode = defaultdict()
    # episode['obs'] = np.ones(shape=(50000, 6, 84, 84)).astype('float32')
    # episode['action'] = np.ones(shape=(50000, 7)).astype('float32')
    # episode['reward'] = np.ones(shape=(50000, )).astype('float32')
    # episode['done'] = np.zeros(shape=(50000, )).astype('float32')
    # buffer.append(episode)
    
    # file = replay_dir / f'{i}.npz'
    # with io.BytesIO() as bs:
    #     np.savez_compressed(bs, **episode)
    #     bs.seek(0)
    #     with file.open('wb') as f:
    #         f.write(bs.read())

time.sleep(100)