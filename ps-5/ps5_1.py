import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os

filepath = os.getcwd()



#the funciton we want to differentiate
def f(x):
    return(1+np.tanh(2*x)/2)

#the jax version of the function we want to differentiate
def f_jax(x):
    return (1+jnp.tanh(2*x)/2)


#analytically calculated derivative of f
def ad(x):
    return(np.power(np.cosh(2*x),-2))

def cd(f,xvals:np.array):
    #set dx
    h=1e-5

    #take the sampel difference
    return ((f(xvals+h/2)-f(xvals-h/2))/h)


x=np.linspace(-2,2,101)
#derivatives using autodiff:

dv_jax= jax.grad(f_jax)
dv=jax.vmap(dv_jax)(x)





fig, ax = plt.subplots(3, figsize=(10, 8))
ax[1].plot(x,ad(x), color='purple', label= 'analytically calculated derivative')
ax[1].scatter(x,cd(f,x), color='red', label='central difference')
ax[0].plot(x,f(x), label='original function')
ax[2].plot(x,ad(x), color='green', label= 'analytically calculated derivative')
ax[2].scatter(x,dv, color='blue', label='jax autodiff')


for i in range (3):
    ax[i].legend()


figname='DifferentiationExample.png'

plt.tight_layout() 
plt.savefig(f'{filepath}/ps5_figs/{figname}')
plt.show()