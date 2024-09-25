import numpy as np
import matplotlib.pyplot as plt
from random import random

print( 'hi im krishna and i love doing homework')

# Constants
N=1000 #number of Tl we start with


TlTau=3.053*60
TlMu=np.log(2)/TlTau


z=np.random.rand(N) #randomly distrbu
tdec=-1*np.log(z)/(TlMu) #an array of the decay time of each of the 10000 atoms
tdec=np.sort(tdec)# sort the array of decay times
NTl = np.arange(999,-1,-1) # create an array that goes from N,N-1,N-0...,1 to represent represent that we start with N atoms at t0, then we have N-1 atoms at t1, then N-2 atoms at t3 etc.. 

plt.plot(tdec,NTl)
plt.title("Tl Decay Over Time")
plt.xlabel("Time (s)")
plt.ylabel("# of Tl atoms")
plt.savefig("ps-3-3.png")




plt.show()