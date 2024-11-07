import numpy as np
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize

filepath=os.getcwd()
filepath='/Users/Krishna/Documents/repositories/phys-ga2000/ps-8'

'''
guess: 
-b0/b1: is the age at which theres a 50-50 chance the person knows bkr
I guess this age is around my age which is 25

1/b1 is the slope. I antincipate a positive slope because younger people havent heard of it, older people have heard it.

b1 should be positive and b0 should be negative, b1 is about -25*b0 


'''

age = np.loadtxt(f'{filepath}/survey.csv', delimiter=',', usecols=0, skiprows=1)  # Age of person surveyed
response = np.loadtxt(f'{filepath}/survey.csv', delimiter=',', usecols=1, skiprows=1)  # Person's response

# logistic function"
def p(x, b0, b1):
    return 1 / (1 + jnp.exp(-(b0 + b1 * x)))

def likelihood(params,x,y):
    b0,b1=params
    px=p(x,b0,b1)
    return np.prod( px**y * (1-px)**(1-y) ) 


# Negative log-likelihood function:
def loglike(params, x, y):
    b0, b1 = params
    px = p(x, b0, b1)  # Probability of response given age
    eps = 1e-12
    nll = -jnp.sum(y * jnp.log(px + eps) + (1 - y) * jnp.log(1 - px + eps))
    return nll

# Initial parameters
start = np.array([-5, 0.2])

# Minimize the negative log-likelihood with a lower tolerance
result = optimize.minimize(loglike, start, args=(age, response), method='BFGS', tol=1e-6)



plt.figure(figsize=(10, 6)) 
model=[p(i,result.x[0],result.x[1]) for i in age]
plt.scatter(age,response, marker='x', color='r', label='data')
plt.scatter(age,model, label='model')
plt.title('Have You Heard of "Be Kind Rewind?')
plt.xlabel("Ages (Years)")
plt.ylabel("Survey Response ")
plt.legend()
plt.savefig(f"{filepath}/figs/fit_start5.png")

plt.show()

h=jax.jacfwd( jax.grad ( loglike ) ) ( result.x, age , response ) #use jax to calculate hessian of b0,b1,age,response

cov= np.linalg.inv(h) #compute the covariance 

standard_errs=np.sqrt(np.diag(cov))

max_likelihood=likelihood(result.x,  age, response)

print(f"for the maximum likelohood: b0={result.x[0]}, b1={result.x[1]}")
print("covariance matrix:")
print(f'the age at which there is a 50-50 shot of knowing is {-result.x[0]/result.x[1]}')
print(cov)
print(f'The maximum likelihood value is {max_likelihood}')
print(f"the standard error for b0={standard_errs[0]}")
print(f"the standard error for b1={standard_errs[1]}" )