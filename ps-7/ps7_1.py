import numpy as np
import matplotlib.pyplot as plt

m_earth=5.974e24 #kg
m_moon=7.348e22 #kg
m_sun=1.988e30
m_jupyter=1.898e27

def f(m,r):
    return( ( 1-r)**2 - m*r- (r**3) * (1-r)**2)
def df(m,r):
    return -2*(1-r)-m + (3*r**2)*(1-r)**2 * 2*(1-r)* r**3

def newton(m,xst=0):
    """
    given the ratio of two masses, return the lagrange point as a ratio fo the distance between them:

    m=M1/M2 where M1 is the smaller mass

    r is the distance between the two masses

    This function uses newton raphson 

    """
    tol = 1.e-9
    maxiter = 1000
    x = xst
    delta=1
    for i in np.arange(maxiter):
        delta = - f(m,x) / df(m,x)
        x=x+delta
        if(abs(delta) < tol):
            return(x)

M_MoonEarth=m_moon/m_earth
M_sunEarth=m_earth/m_sun
m_jupyterSun=m_jupyter/m_sun


R_moon_earth=384400

R_sun_earth=1.486e8 #km
L1=newton(M_MoonEarth)*R_moon_earth 
L2=newton(M_sunEarth)*R_sun_earth
L3=newton(m_jupyterSun)*R_sun_earth

print(f'the lagrange pt between the earth and moon is is :{L1}km from the earth')
print(f'the lagrange pt between the sun and moon is {L2}km')
print(f'the the lagrange point between the a jup mass obj orbitting the sun 1Au from the earth is {L3}km')