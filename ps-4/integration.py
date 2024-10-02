
import numpy as np
from scipy.special import roots_hermite


def gausQuad(f ,a:float,b:float, N:int):
    """
    Inputs
    ______

    f:function 
    some integrand that you want to evaluate with gaussian quadrature
    
    a:float
    the lower bound of the integral

    b:float 
    the upper bound of the integral

    N:int
    the number of points to evaluate the integral

    

    Ouputs
    _______

    I: the evaluated integral



    """
    xp,wp= np.polynomial.legendre.leggauss(N)
    x=np.array( np.float64( (b-a)*xp/2 + (b+a)/2 ) )
    w=np.array( np.float64( .5*(b-a)*wp) )

    I=np.sum( w* f(x) )

    return(I)


def hermQuad(f,N:int):
    '''
    Inputs:
    ______

    f:function 
    some integrand that you want to evaluate with gaussian quadrature

    N:int
    the number of points to evaluate the integral
    
    
    Ouputs:
    _______

    I: the evaluated integral
    '''

    x,w=roots_hermite(N)

    I=np.sum( w*f(x))

    return(I)
