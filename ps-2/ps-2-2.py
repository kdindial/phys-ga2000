import numpy as np


def get_mant(bits):

    m=np.float64(1) # the sum starts at m=1
    for i in range(len(bits)):
        
        m=m+np.float64((bits[i])*2.0**-(i+1))# make sure its float 64

    return(m)

def get_exponent(bits):
    e=0

    for i in range(len(bits)):
        e=e+bits[i]*2**(len(bits)-1-i) #bits[0] is the 2^7 place, bits[1] is the 2^6 place etc...
    e=e-127 #subtract 127 because that is the formula in iee standard

    return(e)

print(np.float32(2**127*(2-2**-23)))
print(np.float64(2**1023*(2-2**-52)))

print("smallest positive:")
print(np.float32(2**-126)*(1+2**-23))
print( np.float64(2**-1023)*(1+2**-52))
