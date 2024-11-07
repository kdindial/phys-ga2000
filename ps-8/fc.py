import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize

data = np.genfromtxt('/Users/Krishna/Documents/repositories/phys-ga2000/ps-8/survey.csv', delimiter=',',skip_header=1)
data=data.T

def log_model(x, beta0, beta1):
    return 1/(1+jnp.exp(-(beta0+beta1*x)))

def negloglike(params, x, data):
    beta0,beta1=params
    m = log_model(x, beta0,beta1)
    lnpi = data*jnp.log(m)+(1-data)*jnp.log(1-m)
    nll = lnpi.sum()
    return -nll

def like(params,ages,answers):
    beta0,beta1=params
    return np.prod(log_model(ages,beta0,beta1)**answers*(1-log_model(ages,beta0,beta1))**(1-answers))


pst = np.array([-5.0, 0.5])

r = optimize.minimize(negloglike, pst, args=(data[0],data[1]), method='BFGS', tol=1e-6)

print(r.x)
model_series=[log_model(i,r.x[0],r.x[1]) for i in data[0]]
plt.plot(data[0],model_series,".",label="Model")
plt.plot(data[0],data[1],".",label="Data")
plt.legend()
plt.xlabel("Ages [Years]")
plt.ylabel("Answers [Yes: 1, No: 0]")
plt.savefig("fit.png")
plt.show()

## Computing the covariance and std variation:
def hessian(params, x, data):
    return jax.jacfwd(jax.grad(negloglike))(params, x, data)


h = hessian(r.x, data[0], data[1])


hess_matrix = hessian(r.x, data[0], data[1])

cov_matrix = np.linalg.inv(hess_matrix)


standard_errors = np.sqrt(np.diag(cov_matrix))
likelihood=like(r.x,data[0],data[1])

print("Estimates for parameters beta0 and beta1:")
print(f"beta0: {r.x[0]}, beta1: {r.x[1]}")
print("\nCovariance Matrix:")
print(cov_matrix)
print("\nStandard Errors:")
print(f"Standard error for beta0: {standard_errors[0]}")
print(f"Standard error for beta1: {standard_errors[1]}")
print(f"Maximum Likelihood is: {likelihood}")