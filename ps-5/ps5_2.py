import numpy as np
import matplotlib.pyplot as plt
import os
filepath = os.getcwd()

"""
Part A
"""
#naive way to write gamma function

def gamma_integrand(a):

    def f(x):
        return(np.power(x,a-1)*np.exp(-x))
    
    return(f)

x=np.linspace(0,5,201)




plt.plot(x,gamma_integrand(2)(x), label='a=2')
plt.plot(x,gamma_integrand(3)(x), label='a=3')
plt.plot(x,gamma_integrand(4)(x), label='a=4')
plt.xlabel('x')
plt.legend()
plt.title(r'Integrand of $\Gamma(a)$ from x=0 to x=5')
plt.savefig(f'{filepath}/ps5_figs/gamma_integrand')
#plt.show()
plt.clf()




"""
Part D
"""

#better way to write gamma function
def gamma_integrand(a):
    def f(x):
        return(np.exp((a - 1) * np.log(x) - x))
    return(f)


x=np.linspace(0,5,101)



'''
Part E

'''



'''
def transformed_gamma_integrand(a):
    c=a-1

    def f(z):
        x = c*z / (1 - z)
        dx_dz = 1*c/ (1 - z)**2
        return np.exp((a - 1) * np.log(x) - x)

    return(f)

plt.plot(x,transformed_gamma_integrand(2)(x), label='a=2')
plt.plot(x,transformed_gamma_integrand(3)(x), label='a=3')
plt.plot(x,transformed_gamma_integrand(4)(x), label='a=4')
plt.xlabel('x')
plt.legend()
plt.title(r'transformed Integrand of $\Gamma(a)$ from x=0 to x=5')
plt.savefig('2a')
plt.show()
plt.clf()




'''




#evaluate gaussian quad


def gamma(a, n_points=20):    
    p,w=np.polynomial.legendre.leggauss(n_points)
    p = 0.5 * (p + 1)
    w = 0.5 * w

    z = p
    c=a-1
    x = z/ (1 - z)
    dx_dz = 1/ (1 - z)**2
    integrand = np.exp((a - 1) * np.log(x) - x + np.log(dx_dz))
    integral = np.sum(w * integrand)

    
    return integral








avals= [3/2, 2,3,5,10]
for a in avals: 
    print(f'computed gamma({a}) is: {gamma(a,20)}')
