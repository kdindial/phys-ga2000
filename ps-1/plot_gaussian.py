import matplotlib.pyplot as plt
import numpy as np

def gausY(x,m,s):
    """returns the output of a guassian the output of the guassian function with standard deviation s, and mean m at x 

    
    Parameters
    ----------
    
    x : float
        input variable


    s : float
        input variable 
        standard deviation

    m : mean
        input variable
        mean of the gaussian
    
    Returns
    -------
    
    val : float
        output variable
        the output of the guassian function with standard deviation s, and mean m at x 
    """
    return np.exp(-((x - m)**2) / (2 * s**2))/ (np.sqrt(2 * np.pi* s**2))
    


x=np.linspace(-10,10,201,True)
mean=0
std=3

y=gausY(x,mean,std)
plt.title('Normal Distribution Function with Standard Dev of 3 and Mean of 0')
plt.plot(x,y,c='green')
plt.savefig('gaussian.png')
