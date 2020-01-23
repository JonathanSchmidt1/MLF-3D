import torch
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image import kernel
from e3nn.image.convolution import SE3Convolution
from e3nn.image import kernel
from datetime import datetime
from functools import partial
import numpy as np
import tracemalloc as tm

tm.start()

grid_size = 90
filter_size = 10

density = np.load(open("../H2_density.npy","rb"),allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
density = torch.tensor(density[96-grid_size//2:96+grid_size//2][:,96-grid_size//2:96+grid_size//2][:,:,99-grid_size//2:99+grid_size//2]).view(1,1, grid_size, grid_size, grid_size).float()

Rs_in = [(1, 0)]  # 1 scalar field
Rs_out = [(1, 0)]  # 1 vector field
radial_window = partial(kernel.gaussian_window_wrapper, mode='conservative', border_dist=0., sigma=.6)
conv = SE3Convolution(Rs_in, Rs_out, size=filter_size, radial_window=radial_window)

with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(10):
        c = conv.float()(density)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
print("max memory used",tracemalloc.get_traced_memory()[1]/1000000000)
