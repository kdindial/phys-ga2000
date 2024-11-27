import numpy as np
cwd='/Users/Krishna/Documents/repositories/phys-ga2000/ps-9'



initial_vals=np.array([0,0,np.cos(np.pi/6)*100,np.sin(np.pi/6)*100], np.float64) # a vector of our initial conditions for x, y, vx and vy

C=0.47 # unitless
p=1.22 #kg/m^3
R=0.08 #m
m=30 #kg
g=9.8 #m/s^2
k=np.power(R,3)*p*C/m #unitless
tprime=np.sqrt(k/R)#t^-1
lprime=1/R



def f(r,t):

    x = r[0]
    y = r[1]
    vx =r[2]
    vy =r[3]


    fx = vx
    fy = vy
    fvx = -np.pi * k * vx * np.sqrt( vx * vx + vy * vy )/2
    fvy = -np.pi * k * vy * np.sqrt( vx * vx + vy * vy )/2 - 1/k
    return( np.array([fx, fy, fvx, fvy]))


def rk4(a:np.float64 , b:np.float64 , N:int , f, r):
    '''
    2nd order rk4 integrator with linear step size

    Inputs:
    a= initial t value
    b= final t value
    N= Number of time steps
    f= 2nd order differential equation that takes in array 'r' and t. 
    r= initial conditions for x and its time derivative v

    Outputs:

    Returns a 5 by N array of x vals, y vals, vx vals, vy vals and t vals



    '''
    #rescale time step:
    a*=tprime
    b*=tprime

    h=(b-a)/N

    #rescale x,y, vx, vy:
    r[0]=r[0]*lprime
    r[1]=r[1]*lprime
    r[2]=r[2]*lprime/tprime
    r[3]=r[3]*lprime/tprime

    tpoints = np.arange(a,b,h)

    xpoints= []
    vxpoints= []
    ypoints= []
    vypoints= []

    for t in tpoints:


        xpoints.append(r[0])

        #lets say that if the cannon ball hits the ground it stops moving in y
        ypoints.append(r[1])

        vxpoints.append(r[2])
        vypoints.append(r[3])
  
        k1= h * f(r , t)
        k2= h * f(r + 0.5 * k1, t + 0.5*h)
        k3= h * f(r + 0.5 * k2, t + 0.5*h)
        k4= h * f(r + k3 , t + h)
        r+= (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    #rescale everything back and return 
    results = np.column_stack((np.array(xpoints)/lprime, np.array(ypoints)/lprime, np.array(vxpoints)*(tprime/lprime), np.array(vypoints)*(tprime/lprime), np.array(tpoints)/tprime))

    return(results)

m1=rk4(a=0.0,b=70.0, N=500.0, f=f,r=initial_vals.copy())
np.savetxt( f'{cwd}/results/mass{m}.csv', m1, delimiter=',', header='x,y,vx,vy,t', comments='', fmt='%f')