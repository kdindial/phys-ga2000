import numpy as np

cwd='/Users/Krishna/Documents/repositories/phys-ga2000/ps-9'


w=1 
r=np.array([1.0,0.0], np.float64) # a vector of our initial conditions for x and v

#function taht returns 
def sho(r,t):
    x=r[0]
    v=r[1]
    fx=v
    fv=-w*w*x
    return( np.array([fx,fv]))


def rk4(a:np.float64 , b:np.float64 , N:np.float64 , f, r):
    '''
    2nd order rk4 integrator with linear step size

    Inputs:
    a= initial t value
    b= final t value
    N= Number of time steps
    f= 2nd order differential equation that takes in array 'r' and t. The array r store the current value of x and its time derivative
    r= initial conditions for x and its time derivative v

    Outputs:

    Returns a 2d 



    '''
    h=(b-a)/N

    tpoints = np.arange(a,b,h)
    xpoints= []
    vpoints= []

    for t in tpoints:
        xpoints.append(r[0])
        vpoints.append(r[1])
        k1= h * f(r , t)
        k2= h * f(r + 0.5 * k1, t+ 0.5*h)
        k3= h * f(r + 0.5*k2, t+0.5*h)
        k4= h * f(r + k3 , t+h)
        r+= (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    results = np.column_stack((xpoints, vpoints, tpoints))

    return(results)

# Save to CSV with column headers

part_a=rk4(a=0.0,b=50.0, N=500.0, f=sho,r=r)
np.savetxt( f'{cwd}/results/8_6_a.csv', part_a, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')

part_b=rk4(a=0.0,b=50.0, N=500.0, f=sho,r=np.array([2.0,0]))
np.savetxt( f'{cwd}/results/8_6_b.csv', part_b, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')



#for part c we change the 2nd order ODE to a non-linear equation. I am going to call it twamp
def aho(r,t):
    x=r[0]
    v=r[1]
    fx=v
    fv=-w*w*x*x*x
    return( np.array([fx,fv]))

aho1=rk4(a=0.0,b=20.0, N=500.0, f=aho,r=np.array([1.0,0]))
aho2=rk4(a=0.0,b=20.0, N=500.0, f=aho,r=np.array([3.0,0]))
aho3=rk4(a=0.0,b=20.0, N=500.0, f=aho,r=np.array([5.0,0]))

np.savetxt( f'{cwd}/results/aho1.csv', aho1, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')
np.savetxt( f'{cwd}/results/aho2.csv', aho2, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')
np.savetxt( f'{cwd}/results/aho3.csv', aho3, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')



def vpo(mu,w):
    def f(r,t):
        x=r[0]
        v=r[1]
        fx=v
        fv= mu * v * (1-np.power(x,2)) -w*w*x
        return( np.array([fx,fv]))
    return f

vpo1= rk4(a=0.0,b=20.0, N=800.0, f=vpo(1,1),r=r)
vpo2= rk4(a=0.0,b=20.0, N=800.0, f=vpo(2,1),r=r)
vpo3= rk4(a=0.0,b=20.0, N=800.0, f=vpo(4,1),r=r)

np.savetxt( f'{cwd}/results/vpo1.csv', vpo1, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')
np.savetxt( f'{cwd}/results/vpo2.csv', vpo2, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')
np.savetxt( f'{cwd}/results/vpo3.csv', vpo3, delimiter=',', header='xpoints,vpoints,tpoints', comments='', fmt='%f')

