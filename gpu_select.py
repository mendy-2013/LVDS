import pynvml

import numpy as np
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
def Auto_Select_GPU():
    pynvml.nvmlInit()
    gpu_list=[]
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_list.append(meminfo.free*100/meminfo.total)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(gpu_list))
    print("共有GPU数量 {}，使用GPU编号：{}, 剩余显存：{:.1f}%".
          format(pynvml.nvmlDeviceGetCount(),np.argmax(gpu_list),np.max(gpu_list)))
    return int(np.argmax(gpu_list))


