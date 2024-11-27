import numpy as np
cwd='/Users/Krishna/Documents/repositories/phys-ga2000/ps-9'



initial_vals=np.array([0,0,np.cos(np.pi/6)*100,np.sin(np.pi/6)*100], np.float64) # a vector of our initial conditions for x, y, vx and vy

C=0.47 # unitless
p=1.22 #kg/m^3
R=0.08 #m
m=0.01 #kg
g=9.8 #m/s^2



def f(r, t):
    x, y, vx, vy = r

    fx = vx
    fy = vy
    drag = -np.pi * R**2 * p * C * np.sqrt(vx**2 + vy**2) / (2 * m)
    fvx = drag * vx / np.sqrt(vx**2 + vy**2)
    fvy = drag * vy / np.sqrt(vx**2 + vy**2) - g

    return np.array([fx, fy, fvx, fvy])


def rk4(a: np.float64, b: np.float64, N: int, f, r):
    """
    4th-order RK integrator with linear step size.

    Inputs:
    a  = initial time value
    b  = final time value
    N  = number of time steps
    f  = function representing the differential equation
    r  = initial state [x, y, vx, vy]

    Outputs:
    Returns an Nx5 array of [x, y, vx, vy, t].
    """
    h = (b - a) / N
    tpoints = np.arange(a, b, h)
    results = np.zeros((len(tpoints), 5))

    for i, t in enumerate(tpoints):
        results[i, :] = [*r, t]
        
        if r[1] < 0:  # Stop if y hits the ground
            results[i:, :] = results[i, :]  # Fill remaining with last state
            break

        k1 = h * f(r, t)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return results

m1=rk4(a=0.0,b=100, N=500.0, f=f,r=initial_vals.copy())
np.savetxt( f'{cwd}/results/mass{m}.csv', m1, delimiter=',', header='x,y,vx,vy,t', comments='', fmt='%f')