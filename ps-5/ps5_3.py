import numpy as np
import matplotlib.pyplot as plt

import os


filepath = os.getcwd()

filepath='/Users/Krishna/Documents/repositories/phys-ga2000/ps-5/'
filename='signal.dat'



'''
Part A: Plot
'''


# Skip the header and load the data
data= np.loadtxt(f'{filepath}/{filename}', delimiter='|', skiprows=1, usecols=(1, 2) ) 

time = np.array(data[:, 0])  # First column: time
signal = np.array(data[:, 1]) # Second column: signal


#Sort the data, I stole this from:
# https://stackoverflow.com/questions/37414916/matplotlib-plotting-in-wrong-order
time, signal=zip(*sorted(zip(time, signal)))




time_mean = np.mean(time)
time_std = np.std(time)
time= (time - time_mean) / time_std






plt.figure(figsize=(10, 8)) 
plt.scatter(time,signal)
plt.xlabel('time ')
plt.ylabel('signal')
plt.title('Signal Data (normalized time from 0-1)')
plt.savefig(f'{filepath}/ps5_figs/3a.png')
plt.clf()



'''
Part B: SVD with 3rd order poly
'''

#construct design matrix
A=np.column_stack([np.ones(time.size), time, time**2, time**3])

#get SVD decomposition
U, W, Vt = np.linalg.svd(A, full_matrices=False)

#get coefficients from svd
W_inv = np.diag(1 / W)

a0,a1,a2,a3=np.dot(Vt.T, np.dot(W_inv, np.dot(U.T, signal)))

#plug in coefficcents to make a 3rd order polynomaial fit
def polyfit3(x):
    return(a0 + a1 * x + a2 * x**2 + a3 * x**3)



# Plot the original data and the polynomial fit
plt.scatter(np.sort(time), signal, label='Original Signal', color='blue' )
plt.plot(time, polyfit3(time), label='3rd Order Poly Fit', color='red')
plt.xlabel('time (normalized)')
plt.ylabel('signal (db)')
plt.savefig(f'{filepath}/ps5_figs/3b.png')
plt.legend()

plt.clf()
'''part c'''

#calculate the residuals:

r=signal-polyfit3(time)
plt.plot(time,r)
plt.title('Residuals Using a 3rd Order Polynomail Fit')
plt.ylabel('r')
plt.xlabel('time')

plt.savefig(f'{filepath}/ps5_figs/3c.png')
#lt.show()
plt.clf()

''' part d'''

#now try using a higher order polynomial

#order of polynomial:

def poly_design_matrix(t, n:int):


    """
    
    Paramaters
    ___________

    t: np.array
    the independent variable of our polynomial fit

    n: int
    The order of the polynomial we are going to fit

    Output
    ________
    
    A: np.ndarray()
    The design matrix. It has dimensions len(t) by o+1 


    """ 
    #make a matrix of all 1s the desired size of the design matrix
    A = np.ones((len(t), n + 1))
    
    # overwrite the values of the design matrix
    for i in range(n+1):
        A[:, i] = np.power(t,i)
        
    return A



def polyfitn(x,n):

    A=poly_design_matrix(time,n)

    condition_number = np.linalg.cond(A)

    # Print the condition number
    print(f"Condition number of the {n}th order polynomial design matrix: {condition_number}")

    U, W, Vt = np.linalg.svd(A, full_matrices=False)

    #get coefficients from svd
    W_inv = np.diag(1 / W)


    coeff=np.dot(Vt.T, np.dot(W_inv, np.dot(U.T, signal)))
    
    k=0
    for i in range(n+1):
        k+=coeff[i]*np.power(x,i)
    return(k)



nvals=np.array([5,10,15,20])

for n in nvals:
    plt.plot(time, polyfitn(time,n), label=f'{n}th Order Poly Fit')

plt.scatter(np.sort(time), signal, label='Original Signal', color='purple', alpha=0.2 )
plt.xlabel('time (normalized)')
plt.ylabel('signal (db)')
plt.legend()
plt.title('Different Order Polynomial Fits of the Signal Data')
plt.savefig(f'{filepath}/ps5_figs/3d.png')

plt.clf()


'''
Part E. Try a trigonmentric fit

'''

def harmonic_design_matrix(t, n:int):

    """
    
    Paramaters
    ___________

    t: np.array
    the independent variable to pass to our harmonic fit

    n: int
    The order of the number of modes we are going to fit

    Output
    ________
    
    A: np.ndarray()
    The design matrix. It has dimensions len(t) by 2n+1


    """ 
    #make a matrix of all 1s the desired size of the design matrix
    A = np.ones((len(t), 2*n + 1))
    T=(np.max(t)-np.min(t))/2
    
    # overwrite the values of the design matrix
    for i in range(n):
            # Cosine terms
            A[:, 2*i+1] = np.cos((i+1)* 2* np.pi * t/T)
            # Sine terms
            A[:, 2*i+2] = np.sin((i+1)* 2* np.pi * t/T)
        
    return A


def harmonic_fit(x,n):

    A=harmonic_design_matrix(time,n)

    condition_number = np.linalg.cond(A)

    # Print the condition number
    print(f"Condition number of harmonic design matrix with {n} modes: {condition_number}")

    U, W, Vt = np.linalg.svd(A, full_matrices=False)

    #get coefficients from svd
    W_inv = np.diag(1 / W)

   


    coeff=np.dot(Vt.T, np.dot(W_inv, np.dot(U.T, signal)))
    print('first 10 ceofficients for the harmonic series:')
    print(coeff[:10])
    T=(np.max(x)-np.min(x))/2
    k=coeff[0]


    for i in range(n):
        k+=coeff[2*i+1]*np.cos(x/T*2*np.pi*(i+1))
        k+=coeff[2*i+2]*np.sin(x/T*2*np.pi*(i+1))
    return(k)



plt.plot(time, harmonic_fit(time,10), label=f'Harmonic fit with {10} modes', color='blue')

plt.plot(time, harmonic_fit(time,200), label=f'Harmonic fit with {200} modes', color='green', alpha=.4)



plt.scatter(np.sort(time), signal, label='Original Signal', color='purple', alpha=.7 )
plt.xlabel('time (normalized)')
plt.ylabel('signal (db)')
plt.legend()
plt.title('Inverse Fourrier Transform of the Signal Data')
plt.savefig(f'{filepath}/ps5_figs/3e.png')
plt.show()



