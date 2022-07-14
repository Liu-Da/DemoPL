import time

import psutil
from tqdm import tqdm

# RAM CPU 利用率 实时统计
# https://stackoverflow.com/a/69511430
with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
    while True:
        rambar.n = psutil.virtual_memory().percent
        cpubar.n = psutil.cpu_percent()
        rambar.refresh()
        cpubar.refresh()
        time.sleep(0.5)
