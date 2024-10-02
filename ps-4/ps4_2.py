import numpy as np
import matplotlib.pyplot as plt
from integration import gausQuad


N=20
m=1

amp=np.linspace(0,2,31)


def V(x):
    """
    The given potential in problem 5.10 is V=x^4
    """
    V=np.power(x,4)
    return (V)


#For a given amplitude a, return the integrand of the integral that gives the period
def f(a):
    def g(x):
        return(np.sqrt(8)/(V(a)-V(x)))


    return(g)

#returns the period with amplitude A
def T(a):
    if a==0:
        return np.inf
    else:
        return(gausQuad(f(a),0,a,N))

plt.figure(figsize=(10, 6)) 

Tvals=[]
for a in amp:
    Tvals.append(T(a))

plt.plot(amp,Tvals)
plt.xlabel('Amplitude (m)')
plt.ylabel('Period (s)')
plt.title('Amplitude vs Period for V=$x^4$')
plt.savefig('ps-4-2.png')
plt.show()

