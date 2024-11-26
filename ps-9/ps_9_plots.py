import matplotlib.pyplot as plt
import numpy as np

path='/Users/Krishna/Documents/repositories/phys-ga2000/ps-9/'

sho = np.loadtxt(f'{path}results/8_6_a.csv', delimiter=',', skiprows=1)

# Split the columns into two separate arrays
sho_x = sho[:,0]
sho_v= sho[:, 1]
sho_t= sho[:, 2]

plt.plot(sho_t,sho_x,label="initial x=1")
plt.ylabel('displacement (m)')
plt.xlabel('time elapsed (s)')
plt.title("simple harmonic oscilator with runge kutta")
plt.savefig(f'{path}/figs/8_6_a.png')


sho_b=np.loadtxt(f'{path}results/8_6_b.csv', delimiter=',', skiprows=1)
sho_b_x = sho_b[:,0]
sho_b_v= sho_b[:, 1]
sho_b_t= sho_b[:, 2]

plt.plot(sho_b_t, sho_b_x,label="intial x=2")
plt.legend()
plt.ylabel('displacement (m)')
plt.xlabel('time elapsed (s)')
plt.title("simple harmonic oscilator with two different intial amplitude")
plt.savefig(f'{path}/figs/8_6_b.png')

plt.clf()

aho1= np.loadtxt(f'{path}/results/aho1.csv', delimiter=',', skiprows=1)
aho2= np.loadtxt(f'{path}/results/aho2.csv', delimiter=',', skiprows=1)
aho3= np.loadtxt(f'{path}/results/aho3.csv', delimiter=',', skiprows=1)

aho=np.array([aho1,aho2,aho3])

for i in range (len(aho)):
    plt.plot(aho[i][:,2],aho[i][:,0],label=f'initial x={aho[i][0,0]}')

plt.ylabel('displacement (m)')
plt.xlabel('time elapsed (s)')
plt.legend()
plt.title("anharmonic oscilator with different intial conditions")
plt.savefig(f'{path}/figs/aho.png')

plt.clf()

#now we need to make phase diagrams

plt.plot(sho_b_x, sho_b_v,label="intial x=2")
plt.plot(sho_x, sho_v,label="intial x=1")
plt.legend()
plt.ylabel('displacement (m)')
plt.xlabel('velocity (m/s)')
plt.title("SHO Phase Space Plot")
plt.savefig(f'{path}/figs/sho_phase.png')

plt.clf()

for i in range (len(aho)):
    plt.plot(aho[i][:,0],aho[i][:,1],label=f'initial x={aho[i][0,0]}')

plt.legend()
plt.ylabel('displacement (m)')
plt.xlabel('velocity (m/s)')
plt.title("Anharmonic Oscilator Phase Space Diagram")
plt.savefig(f'{path}/figs/aho_phase.png')

plt.clf()

vpo1 = np.loadtxt(f'{path}/results/vpo1.csv', delimiter=',', skiprows=1)
vpo2 = np.loadtxt(f'{path}/results/vpo2.csv', delimiter=',', skiprows=1)
vpo3 = np.loadtxt(f'{path}/results/vpo3.csv', delimiter=',', skiprows=1)


mu=[1,2,4]
vpo=[vpo1,vpo2,vpo3]
for i in range (len(vpo)):
    plt.plot(vpo[i][:,0],vpo[i][:,1],label=f'mu={mu[i]}, w=1')

plt.legend()
plt.ylabel('displacement (m)')
plt.xlabel('velocity (m/s)')
plt.title("Van der Pol Oscillator Oscilator Phase Space Diagram")
plt.savefig(f'{path}/figs/vdp_phase.png')

plt.clf()

#now exercise 8.7

m1 = np.loadtxt(f'{path}/results/mass1.csv', delimiter=',', skiprows=1)
m2 = np.loadtxt(f'{path}/results/mass5.csv', delimiter=',', skiprows=1)
m3 = np.loadtxt(f'{path}/results/mass10.csv', delimiter=',', skiprows=1)


trajectories=[m1,m2,m3]
masses=[1,5,10]
for i in range (len(trajectories)):
    plt.plot(trajectories[i][:,0],trajectories[i][:,1],label=f'mass={masses[i]}kg')
plt.legend()
plt.xlabel('x position (m))')
plt.ylabel('y position (m)')
plt.title("Cannon Ball Trajectory with Air Resistance")
plt.savefig(f'{path}/figs/trajectories.png')



