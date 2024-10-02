import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from integration import gausQuad


V=0.001 #cubic meters
p=6.022*10**28 #meters^-3
theta=428 #K
kb=1.380649*10**-23 # joule per kelvin

T=100
a=0 #bottom bounds of integration
b=theta/T #top bounds of integration
#To do guassian quadrature, we need w_p (the weights) and x_p which are the x calues we are going to calculate
#Then we need to remap the w_p and x_p

N=50




def f(x):
    f=np.power(x,4)*np.exp(x)/np.square((np.exp(x)) -1)
    return(f)




"""
PART A
"""

def cv(T,N):
    a=0
    b=theta/T

    I=gausQuad(f,a,b,N)
    #multiply integral by some constan
    C=9*V*p*kb*(T/theta)**3
   
    return(C*I)

T=np.arange(5,500,5)
"""
PART B
"""

#make an array to store the cv values
C=[]


for t in T :
    C.append(cv(t,50))

plt.plot(T,C)
plt.xlabel( 'Temperature (K)')
plt.ylabel('Specific Heat ( J/(kg * K )')
plt.title('Evaluating Specific Heat using Guassian Quad')

plt.figure(figsize=(10, 6)) 
plt.savefig('ps-3-1b')
plt.clf()

Nvals=np.arange(10,71,10)

T=5
C=[]
for n in Nvals:
    C.append(cv(T,n))


plt.scatter(Nvals,C)
plt.xlabel( ' N ' )
plt.ylabel('Specific Heat ( J/(kg * K )')
plt.title('Evaluating Specific Heat Using Guassian Quad for T=5K')
plt.savefig('ps-3-1c')
#plt.show()


