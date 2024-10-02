import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import roots_hermite
from typing import Callable
from integration import gausQuad, hermQuad

#I copied this function from your recitation. I would not have been able to come up with it by myself. 
# I did not know that I could make a function that returns a function. Very cool.
def Herm(n: int):
    if n < 0: 
        raise ValueError("n<0")
    if n == 0:
        return lambda x: np.ones(x.shape)
    elif n == 1:
        return lambda x: 2 * x
    else: 
        def f(x):
            # Recursive calculation of the Hermite polynomial
            return 2 * x * Herm(n - 1)(x) - 2 * (n - 1) * Herm(n - 2)(x)
        return f

xvals = np.linspace(-4, 4, 201)

def psi(n):
    def p(x):
        H=Herm(n)(x) #hermite polynomial n
        e=np.exp(-x**2/2) #gaussian 
        norm=1 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi)) #normalization

        return  (e*norm*H) # Return the result
    return p


plt.figure(figsize=(10, 6)) 

for i in range(5):


    plt.plot( xvals,psi(i)(xvals),label=( fr'$\psi_{i}$') )

plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.savefig('psi_1-4.png')

plt.clf()

i=30
xvals=np.linspace(-10,10,51)
plt.plot(xvals, psi(i)(xvals) )
plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.title('$\psi_{30}(x)$')
plt.savefig('psi_30.png')

plt.clf()


#returns the a function that takes the norm of the psi(n)
# inputs an n and outputs a function of x
def norm(n):
    def f(x):
        p=psi(n)(x)
        return(p*p)
    return(f)

##returns the integrand of <x^2> for psi_n
#input: n x  
#output: a function of x
def ms(n):
    def f(x):
        p=psi(n)(x)
        return( p*p*x*x )
    return(f)

#remaps the integrand of <x^2> of psi_n(x) from -inf to inf --> -pi/2 to pi/2 
#input:a function of n  
# output a function of x
def msRemapped(n):
    def f(x):

        #go from f(x) to f(tanx)
        z=np.tan(x)
        #go adjust dx accordingly
        dz=1/(np.cos(x)**2)

        k=ms(n)(z)*dz
        return( k)
    return(f)

#remaps the integrand of <psi_n)^2> from -inf to inf --> -pi/2 to pi/2 
#inputs a function of n and outputs a function of x
def normRemapped(n):
     def f(x):
        z=np.tan(x)
        dz=1/(np.cos(x)*np.cos(x))

        k=norm(n)(z)*dz
        return( k )
     return(f)


N=100
#take the square root of <x^2>
hermRms5=np.sqrt( hermQuad(ms(5),N) )

hermNorm5=hermQuad(norm(5),N)


a=-np.pi
b=np.pi
gausNorm5=gausQuad(normRemapped(5),a,b,N)

gausRms5=np.sqrt(gausQuad(msRemapped(5),a,b,N))

xvals=np.linspace(-4,4,41)

for i in range(5):

    plt.plot( xvals,norm(i)(xvals),label=( fr'$\psi_{i}$') )

plt.legend()
plt.title(r'$\psi_n(x)^2$')
plt.xlabel('x')
plt.ylabel(r'$\psi(x)^2$')

plt.savefig('psi_1-4_norm.png')

plt.clf()



for i in range(5):
    plt.plot( xvals,ms(i)(xvals),label=( fr'$\psi_{i}$') )

plt.legend()
plt.title(r'$x^2\psi_n(x)$')
plt.xlabel('x')
plt.ylabel(r'$x^2\psi(x)$')

plt.savefig('psi_1-4_rms.png')



print(f'The norm of psi5 using gaus poly N=100 is:{gausNorm5}')
print(f'The norm of psi5 using gaus herm N=100 is: {hermNorm5}')
print(f'The rms of psi5 using gaus herm N=100 is: {gausRms5}')
print(f'The rms using herm quad  N=100is {hermRms5}')