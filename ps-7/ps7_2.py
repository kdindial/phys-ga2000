import math
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize
jax.config.update("jax_enable_x64", True)


def parabolic_step(func=None, a=None, b=None, c=None):
    """returns the minimum of a function as approximated by a parabola"""
    fa = func(a)
    fb = func(b)
    fc = func(c)

    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)

    #If singular, return b 
    if(np.abs(denom) < 1.e-15):
        x=b
    else:
        x = b -.5*numer/denom

    return(x)



def parabolic_minimize(func=None, astart=None, bstart=None, cstart=None,
                       tol=1.e-5, maxiter=10000):

    ''' 

    Uses the parabolic step function to minimize a 1D function. 
    First it fits 3 points f(a), f(b), f(c) to a prabola, then it fits it to a parabola.
    If the parabola min is greater than c, it shifts the brackets



    '''
    a = astart
    b = bstart
    c = cstart
    bold = b + 2. * tol
    niter = 0
    while((np.abs(bold - b) > tol) & (niter < maxiter)):
        bold = b
        b = parabolic_step(func=func, a=a, b=b, c=c)
        if(b < bold):
            c = bold
        else:
            a = bold
    return(b)

def golden(func=None, astart=None, bstart=None, cstart=None, tol=1.e-8):
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    while(np.abs(c - a) > tol):
        # Split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        step = np.array([b, x])
        fb = func(b)
        fx = func(x)
        if(fb < fx):
            c = x
        else:
            a = b
            b = x 
    return(b)
def y(x):
    return (x-0.3)**2 * np.exp(x)

'''
start with brackets a and c. Then pick b to be midway between them. 
Do parabolic fit and check:
did the parabolic step fall outside of the bracketing interval (in parabolic step this is "b")

is the parabolic step greater than the last?. If this fails then revert to golden section search

'''

def brent1D(func,a,b,tol=1e-9, maxiter=100):
    '''
    Params:

    f: function that we are minimizing
    a: left bracket a must be less than c
    b: right bracket
    tol: tolerance 
    max_iter: max number of iterations

    returns: xmin
    '''

    #at any given time, keep track of four points x0,x1,x2,x3:
    gsection = (3. - np.sqrt(5)) / 2

    x = w = v = a+ gsection*(b-a)
    fw = fv = fx =func(x)
    zepp=1e-10


    d = b-a # distance moved since last step
    e=d #distance moved in step before last


    for i in range(maxiter): #main program loop

        xm=0.5*(a+b)#idk this step is in numerical recipes
        tol1= tol* np.abs(x)+zepp
        tol2=2.0*tol1

        if abs(x-xm) <= ( tol2 - 0.5 * (b-a)):# test to see if within tolerance of finding minimum

            fmin=func(x)
            return(x,fx)

        if (abs(e)>tol1):

            #parabolic fit I stole from numerical recipes
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                # Accept parabolic interpolation
                d = p / q
                u = x + d
        
                if (u - a) < tol2 or b - u < tol2:
                    #i dont really understand this step but I am taking it from numerical recipes
                    d = tol1 if xm - x >= 0 else -tol1
            else:
                # Use golden-section
                d = gsection * (b - x) if x < xm else -gsection * (x - a)
        else:
            # Use golden-section
            d = gsection * (b - x) if x < xm else -gsection * (x - a)
        
        u = x + d if abs(d) >= tol1 else x + (tol1 if d > 0 else -tol1)
        fu=func(u)

        # now update everything

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, func(x)





        


                

brent_x,brent_y=brent1D(y,-1,1)


print(f'using my implementation of brents method, the min is :{brent_y} at x={brent_x}')

scipyOptimizer= optimize.brent( y, brack=None, tol=1.48e-08, full_output=1)
scipy_x,fval,iter,funcalls= scipyOptimizer
print(f'using scipy.optimize.brent, the min is :{y(scipy_x)} at x={scipy_x}')
