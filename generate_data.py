import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
import torch
from datetime import datetime
import pickle

atoms = 'H 0.00000000 0.0000000000 -0.3699999734101;H 0.00000000 0.0000000000  0.3699999734101'
basis = {'H': gto.basis.parse('''
H    S
  0.300996E+04           0.994949E-04
  0.451373E+03           0.772617E-03
  0.102806E+03           0.405930E-02
  0.291474E+02           0.170977E-01
  0.952878E+01           0.615690E-01
  0.345638E+01           0.192015E+00
  0.135638E+01           0.500000E+00
H    S
  0.561300E+00           1.0000000
H    S
  0.241026E+00           1.0000000
H    S
  0.104273E+00           1.0000000
H    S
  0.439734E-01           1.0000000
H    S
  0.171020E-01           1.0000000
H    P
  0.180000E+03           0.352640E-01
  0.300000E+02           0.500000E+00
H    P
  0.660000E+01           1.0000000
H    P
  0.210000E+01           1.0000000
H    P
  0.950000E+00           1.0000000
H    P
  0.450000E+00           1.0000000
H    P
  0.150000E+00           1.0000000
H    D
  0.210000E+01           1.0000000
H    D
  0.700000E+00           1.0000000
H    D
  0.300000E+00           1.0000000
  ''')}
mol = gto.M(atom=atoms, basis=basis)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()
dm = mf.make_rdm1()
size = 18
# Use default mesh grids and weights
grid = np.linspace(-size//2,size//2, int(size/0.2+1))
coords =  np.hstack((np.meshgrid(grid,grid,grid)[0].flatten().reshape((int(size/0.2+1)**3,1)),
     np.meshgrid(grid,grid,grid)[1].flatten().reshape((int(size/0.2+1)**3,1)),
                                                      np.meshgrid(grid,grid,grid)[2].flatten().reshape((int(size/0.2+1)**3,1))))
# weights = mf.grids.weights
# weights_list.append(weights)
ao_value = numint.eval_ao(mol, coords, deriv=1)
# The first row of rho is electron density, the rest three rows are electron
# density gradients which are needed for GGA functional
rho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
print(rho.shape)
exc, vxc = dft.libxc.eval_xc( 'b3lyp', rho)[:2]
print('Exc = %.12f' % np.einsum('i,i->', exc, rho[0]/125.0))
Exc = np.einsum('i,i->', exc, rho[0]/125.0)
np.save(open("../rho_b3lyp.npy","rb"),rho[0])
np.save(open("../vxc_b3lyp.npy","rb"),vxc)
np.save(open("../exc_b3lyp.npy","rb"),exc)
np.save(open("../Exc_b3lyp.npy","rb"),Exc)

# coords = mf.grids.coords
# weights = mf.grids.weights
# ao_value = numint.eval_ao(mol, coords, deriv=1)
# # The first row of rho is electron density, the rest three rows are electron
# # density gradients which are needed for GGA functional
# rho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
# print(rho.shape)
#
# #
# # Evaluate XC functional one by one.
# # Note: to evaluate only correlation functional, put ',' before the functional name
# #
#
# #
# # Evaluate XC functional together
# #
# exc, vxc = dft.libxc.eval_xc('b3lyp', rho)[:2]
# print('Exc = %.12f' % np.einsum('i,i,i->', exc, rho[0], weights))
