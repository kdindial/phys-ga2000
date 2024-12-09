import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

path='/Users/Krishna/Documents/repositories/phys-ga2000/ps-10/'


m=9.109e-31 #Mass of electron in kg
L=1e-8 #Length of box in m
x0=L/2
sig=1e-10
kap=5e10
N=1000
x_vals=np.linspace(0,L,N)
hbar=1.054e-34 # hbar in Js
dt=1e-18
frames=5
tsteps=1000
# Initial wavefunction
psi_t0 = np.exp(-(x_vals - x0) ** 2 / (2 * sig ** 2)) * np.exp(-1j * kap * x_vals)

# Fourier coefficients of the initial wavefunction
psi_k_t0 = fft(psi_t0)

# Wavenumbers
k_array = np.fft.fftfreq(N, d=x_vals[1] - x_vals[0])
k_array = 2 * np.pi * k_array  # Convert to angular wavenumbers

# Dispersion relation: Energy of modes
omega_k = (hbar * k_array ** 2) / (2 * m)

# Time evolution and plotting
big_step = tsteps / frames
for i in range(frames):
    t = i * big_step * dt

    # Time evolution in Fourier space
    psi_k_t = psi_k_t0 * np.exp(-1j * omega_k * t)

    # Transform back to spatial domain
    psi_t = ifft(psi_k_t)

    # Plot the real part of the wavefunction
    plt.plot(x_vals * 1e9, np.real(psi_t), label='Re(ψx)')
    plt.xlabel('X (nm)')
    plt.ylabel('Real(ψx) un_normalized)')
    plt.title(f'Wavefunction at t = {t} s')
    plt.savefig(f'{path}/figs/Spectral_method_{i * big_step}_steps.png')
    plt.clf()
    
    plt.scatter(k_array*1e-9, np.real(psi_t), label='Re(ψk)')
    plt.xlabel('k (1/nm)')
    plt.ylabel('Real(ψ(k)) un normalized )')
    plt.title(f'Wavefunction at t = {t} s')
    plt.savefig(f'{path}/figs/Spectral_method__kspace_{i * big_step}_steps.png')
    plt.clf()

