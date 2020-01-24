import torch
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image import kernel
from e3nn.image.convolution import SE3Convolution
from e3nn.image import kernel
from datetime import datetime
from functools import partial
import numpy as np
import tracemalloc as tm
from e3nn.non_linearities.activation import Activation
import torch.nn as nn


tm.start()

grid_size = 90
filter_size = [5,2]
Rs_in = [(1, 0)]  # 1 scalar field
Rs_out = [[(2, 0)],[(1,0)]]
radial_window = partial(kernel.gaussian_window_wrapper, mode='conservative', border_dist=0., sigma=.6)
#conv = SE3Convolution(Rs_in, Rs_out, size=filter_size, radial_window=radial_window)
ELU_2 = Activation((2,nn.ELU()))
ELU_1 = Activation((2,nn.ELU()))

nn.Sequential(
SE3Convolution(Rs_in, Rs_out[0], size=filter_size[0], radial_window=radial_window),
ELU_2,
SE3Convolution(Rs_out[0], Rs_out[1], size=filter_size[0], radial_window=radial_window),
ELU_1,
SE3Convolution(Rs_out[0], Rs_out[1], size=filter_size[1], radial_window=radial_window),
)

optimizer = torch.optim.Adam(conv.parameters(), lr=0.01)


density_system = np.load(open("../H2_density.npy","rb"),allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
vxc_system = np.load(open("vxc.npy","rb"), allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
density = torch.tensor(density_system[96-grid_size//2:96+grid_size//2][:,96-grid_size//2:96+grid_size//2][:,:,99-grid_size//2:99+grid_size//2], requires_grad=True).view(1,1, grid_size, grid_size, grid_size).float()
vxc = torch.tensor(vxc_system[96-grid_size//2:96+grid_size//2][:,96-grid_size//2:96+grid_size//2][:,:,99-grid_size//2:99+grid_size//2]).view(1,1, grid_size, grid_size, grid_size).float()
del density_system
del vxc_system


loss_f = torch.nn.MSELoss()

with torch.autograd.profiler.profile(use_cuda=False) as prof:
	for i in range(1000):
		c = conv.float()(density)
		E = torch.sum(c)
		potentials_nn, = torch.autograd.grad(E, density,
                       grad_outputs = E.data.new(E.shape).fill_(1), retain_graph = True,
                       create_graph=True)
		loss_p = loss_f(potentials_nn*125.0, vxc)
		loss_int = loss_f(torch.sum(potentials_nn*density), torch.sum(vxc*density)/125)
		loss = loss_p+loss_int
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(loss, loss_p, loss_int)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
print("max memory used",tm.get_traced_memory()[1]/1000000000)
print("allocated memory",tm.get_traced_memory()[0]/1000000000)

