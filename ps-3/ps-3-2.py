import numpy as np
import matplotlib.pyplot as plt
from random import random

print( 'hi im krishna and i love doing homework')

# Constants
NBi0=10000 #number of Bi209 atoms. Im going to call it Bi0
NTl=0 #number of Tl atoms
NTi= 0 #number of thal
NPb = 0 #number of lead atoms
NBi1= 0 #number of Bi209 atoms. Im going to call it Bi1
BiTau= 46*60
TlTau= 2*60 #half life of Tl
PbTau= 3.3*60 #half life of Pb to Bi
h = 1.0 # size of time step in seconds
tmax=20000

pBi= np.float32(1-2**(-h/BiTau)) #probability Bi213 decays
pTl= np.float32(1-2**(-h/TlTau)) #probability Tl decays
pPb= np.float32(1-2**(-h/PbTau)) #)probability Pb decays

pBiPb=0.9791 #probability Bi decays to Pb
""" 
p = 1-2(-h/tau)

"""

tpoints = np.arange(0.0, tmax, h)

Bi1points=[]
Pbpoints=[]
Tlpoints=[]
Bi0points=[]

#Pb to Bi209 loop

for t in tpoints:
    Pbpoints.append(NPb)
    Bi1points.append(NBi1)
    Bi0points.append(NBi0)
    Tlpoints.append(NTl)

    #simulate the number of attoms that decay in this time step

    Pbdecay = 0
    for i in range(NPb): #for each Pb atom decide if it decays
        if random()<pPb: 
            Pbdecay +=1

    # adjust the number of each atom according to the number of decays we calculated 
    NPb -= Pbdecay
    NBi1 += Pbdecay


    # sim the number of Tl atoms that decay to Pb in this time step
    Tldecay=0
    for i in range(NTl):

        if random()<pTl:
            Tldecay+=1
        #adjust the number of each aotm according to the number of atoms that decayed in this time step
    NPb += Tldecay
    NTl -= Tldecay

    #now we have to sim the number of Bi213 atoms that decay in this time step
    BiPbdecay=0
    BiTldecay=0

    

    for i in range(NBi0):
        if random()<pBi:
            #now simulate if the Bi213 will decay into Tl or Pb
            if random()<pBiPb:
                BiPbdecay+=1
            else: 
                BiTldecay+=1
    NTl+=BiTldecay
    NPb+=BiPbdecay
    NBi0-=BiTldecay
    NBi0-=BiPbdecay



"""
print("b1")         
print(len(Bi1points))
print("pb")
print(len(Pbpoints))
print("Tl")
print(len(Tlpoints))
print("Bi0")
print(len(Bi0points))
print("t")
print(len(tpoints))


"""

plt.plot(tpoints,Pbpoints,label='Pb')
plt.plot(tpoints,Bi1points,label='Bi209')
plt.plot(tpoints,Tlpoints, label='Tl')
plt.plot(tpoints,Bi0points, label='Bi213')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("# Atoms")
plt.title("Decay of Bi213 atoms")
plt.savefig('ps-3-2.png')
#plt.show()





# Lists of plot points
# Number of thallium atoms # Number of lead atoms
# Half l i f e of thallium in seconds ‡ Size of time-ster in seconds
# Probability of decav in one step # Total time