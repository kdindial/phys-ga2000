import numpy as np
import scipy.linalg
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import animation
path='/Users/Krishna/Documents/repositories/phys-ga2000/ps-10/'

dt= 1e-18
hbar = 1.0546e-36
L = 1e-8
m = 9.109e-31
N = 1000 # Grid slices
x0=L/2
sigma=1e-10 #meters 
kappa=5e10 #m^-1


dx=L/N


#i copied and pasted this straight from https://nbviewer.org/github/blanton144/computational-grad/blob/main/docs/notebooks/pde.ipynb
#Thank you blanton
def diffusion_cn_step(q=None, dt=None, dx=None ):
    #change alpha for the schrodinger equation
    alpha= dt * 1j * hbar / ( 4 * m * dx * dx)

    diag = (1. + 2 * alpha) * np.ones(len(q))
    offdiag = (- alpha) * np.ones(len(q))
    sparseA = np.zeros((3, len(q)), dtype=complex)


    sparseA[0, :] = offdiag
    sparseA[1, :] = diag
    sparseA[2, :] = offdiag
    qhalf = q 
    qhalf[1:-1] = q[1:-1] + alpha * (q[2:] - 2. * q[1:-1] + q[:-2])
    qnew = scipy.linalg.solve_banded((1, 1), sparseA, qhalf)
    return(qnew)



#define our wave equation at t=0 as a function of x
def psi_func(x):
	return np.exp(-(x-x0)**2/2/sigma**2)*np.exp(1j*kappa*x)

#define x vals
xvals=np.linspace(0,L,N+1)
dx=L/(N) #grid size
#make a wave vector at t=0 from 0 to L
psi_t0= np.zeros(N+1, complex)
psi_t0[:]=psi_func(xvals)
psi_t0[0]=0
psi_t0[N]=0
#perform one time step
psi_t1=diffusion_cn_step(psi_t0,dt,dx)
#peform another time step
psi_t2=diffusion_cn_step(psi_t1,dt,dx)


#perform time steps:
tsteps=int(200000)
PSI = np.zeros((tsteps + 1, N + 1), dtype=complex)
PSI[0,:]=psi_t0


for i in range(tsteps):
    PSI[i+1,:]=diffusion_cn_step(PSI[i], dt, dx)

frames=5

big_step=tsteps/frames
print(frames)
for i in range(frames):

    plt.plot ( xvals*1e9, PSI[i*int(big_step),:])
    plt.xlabel('X (nm)')
    plt.ylabel('Real(PSI) unnormalized')
    plt.title(f'PSI(X) after {i*big_step*dt:.2e} s')
    plt.savefig(f'{path}/figs/Crank_Dat_{i*big_step}_steps.png')
    plt.clf()





