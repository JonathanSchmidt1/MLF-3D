import torch
from se3cnn.image.convolution import SE3Convolution
from se3cnn.image import kernel
from e3nn.image.convolution import SE3Convolution
from e3nn.image import kernel
from datetime import datetime
from functools import partial
import numpy as np
import tracemalloc as tm
from e3nn.non_linearities.scalar_activation import ScalarActivation
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)

tm.start()

grid_size = 90
filter_size = [11,5]
Rs_in = [(1, 0)]  # 1 scalar field
Rs_out = [[(2, 0)],[(1,0)]]
radial_window = partial(kernel.gaussian_window_wrapper, mode='conservative', border_dist=0., sigma=.6)
#conv = SE3Convolution(Rs_in, Rs_out, size=filter_size, radial_window=radial_window)
ELU_1 = ScalarActivation([(1,nn.ELU())])
ELU_2 = ScalarActivation([(2,nn.ELU())])


Functional = nn.Sequential(
SE3Convolution(Rs_in, Rs_out[0], size=filter_size[0], radial_window=radial_window, padding=5),
ELU_2,
SE3Convolution(Rs_out[0], Rs_out[1], size=filter_size[0], radial_window=radial_window, padding=5),
ELU_1,
SE3Convolution(Rs_out[1], Rs_out[1], size=filter_size[1], radial_window=radial_window, padding=2),
)
Functional = Functional
optimizer = torch.optim.Adam(Functional.parameters(), lr=0.0001)


#density_system = np.load(open("../H2_density.npy","rb"),allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
#vxc_system = np.load(open("vxc.npy","rb"), allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
density_system = np.load(open("../rho_b3lyp.npy","rb"),allow_pickle=True).reshape(191,191,191) #center of molecule [96][96][99]
vxc_system = np.load(open("../vxc_b3lyp.npy","rb"), allow_pickle=True)[0].reshape(191,191,191) #center of molecule [96][96][99]
Exc = torch.tensor(np.load(open("../Exc_b3lyp.npy","rb"), allow_pickle=True))
density = density_system[96-grid_size//2:96+grid_size//2][:,96-grid_size//2:96+grid_size//2][:,:,99-grid_size//2:99+grid_size//2]
density = torch.tensor(density, requires_grad=True).view(1,1, grid_size, grid_size, grid_size)
vxc = torch.tensor(vxc_system[96-grid_size//2:96+grid_size//2][:,96-grid_size//2:96+grid_size//2][:,:,99-grid_size//2:99+grid_size//2]).view(1,1, grid_size, grid_size, grid_size)
print(vxc.shape)
print(torch.sum(vxc*density)/125.0)
del density_system
del vxc_system


loss_f = torch.nn.MSELoss()

with torch.autograd.profiler.profile(use_cuda=False) as prof:
	for i in range(100):
		c = Functional(density)
		print(c.shape)
		E = torch.einsum("i,i->",c.flatten(),density.flatten())/125.0
		potentials_nn, = torch.autograd.grad(E, density,
                       grad_outputs = E.data.new(E.shape).fill_(1), retain_graph = True,
                       create_graph=True)
		loss_E = loss_f(Exc,E)
		loss_p = loss_f(potentials_nn*125.0, vxc)
		loss_int = loss_f(torch.sum(potentials_nn*density), torch.sum(vxc*density)/125)
		loss = loss_E
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("total loss %10.5E" % loss, "v loss %10.2E" % loss_p, "int loss %10.2E" % loss_int, "E loss %10.5E" % loss_E)

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
print("max memory used",tm.get_traced_memory()[1]/1000000000)
print("allocated memory",tm.get_traced_memory()[0]/1000000000)


