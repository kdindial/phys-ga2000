import numpy as np
import matplotlib.pyplot as plt

Nvals=np.arange(1,502,10) # number of x samples 


m=1000 # number of y samples

yvals=[]
avgs=np.array([])
vars=np.array([])
kurts=np.array([])
skews=np.array([])
def kurt(x):
    """
    calculates the kurtosis of an array of values. the input

    inputs:
    _________
    x:np.array 
    The array of numbers which we want to calculate the kurtosis of 1d numpy array

    outputs:
    k:float32 
    the kurtosis of x
    """

    k= np.mean( (x-np.mean(x))**4 ) / (np.var(x))**2-3
    return(k)

def skew(x):
  """
    calculates the skew of an array of values

    inputs:
    _________
    x:np.array 
    The array of numbers which we want to calculate the skew of 1d numpy array

    outputs:
    k:float32 
    the skew of x
    """
  n=len(x)
  s=(n / ((n - 1) * (n - 2))) * np.sum(((x - np.mean(x)) / np.std(x)) ** 3)
  return(s)

for i in range(Nvals.size):
    #I was originally going to do nested for loop, where for each n value, I generate n x values to make a y value. I loop over this m times to make m y values for each N value. When working with Taha, I saw that I could avoid the second for loop by by making an m by N array. Each element m contains an array of N xvals that will be collapsed into m yvals.
    xvals=np.random.exponential(size=(m,Nvals[i])) #make an array that contains m arrays. Each array has N elements where N is the particular N value that we are currently itterating over
    y=np.array(np.mean(xvals, axis=1)) # generate our yvals by taking the avging over each array of Nvals. 
    
    avgs=np.append(avgs,np.mean(y))
    vars=np.append(vars,np.var(y))
    kurts=np.append(kurts,kurt(y))
    skews=np.append(skews,skew(y))
    yvals.append(y) # add each y value we generate to our array of y vals for a given N size





fig, ax = plt.subplots(2,2)
ax[0,0].plot(Nvals,avgs)
ax[0,0].set_ylabel('Mean')

ax[0,1].set_ylabel('Variance')
ax[0,1].plot(Nvals,vars)

ax[1,0].set_ylabel('Skew')
ax[1,0].plot(Nvals,skews)

ax[1,1].plot(Nvals,kurts)
ax[1,1].set_ylabel('Kurtosis')

plt.subplots_adjust(hspace=.5,wspace=.5)

ax[0,0].set_xlabel('# of N vals')
ax[0,1].set_xlabel('# of N vals')
ax[1,0].set_xlabel('# of N vals')
ax[1,1].set_xlabel('# of N vals')


plt.savefig('momentsVsN.png')


plt.clf()

for i in range(len(Nvals)):
    plt.title(f'Histogram with N={Nvals[i]}')
    plt.hist(yvals[i], bins=50, label=f'N = {Nvals[i]}')
    plt.savefig(f'N={Nvals[i]}.png')
    plt.clf()

# Comparing with the theoretical Guassian
