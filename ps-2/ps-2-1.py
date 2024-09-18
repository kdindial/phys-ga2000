import numpy as np

#I am going to steal Michael Blanton's "get bits" function to 

def get_bits(number):
    """ For a Numpy quantity, return bit representation

    Inputs:
    -----
    number: Numpy Value

    Returns:
    ----
    bits : list

        list of binary values, highest to lowest significance
    """

    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))


x=np.float32(100.98763)
ex=get_bits(x)[1:9]
frac=get_bits(x)[9:32]
sign=get_bits(x)[0]
print(x.tobytes())
print("the sign is "+ str(sign))
print("exponent is: " + str(ex))
print("the mantissa is: "+ str(frac))

mbits=get_bits(x)[9:32] #get the bits for the mantissa


def get_exponent(bits):
    e=0

    for i in range(len(bits)):
        e=e+bits[i]*2**(len(bits)-1-i) #bits[0] is the 2^7 place, bits[1] is the 2^6 place etc...
    e=e-127 #subtract 127 because that is the formula in iee standard

    return(e)

e=get_exponent(ex)


#im going to express the matnissa as a fraction of some numerator over some denomenator
#we know the least common denominator of the mantissa is 2^-23
#we can write the fraction sum as 2^(22-i)/2^23
def get_num(bits):
    n=0
    for i in range(len(bits)):
        n=n+bits[i]*2**(22-i)
    return(n)

numerator=get_num(frac)+2**23 #calculate the sum of the numerator and then add "1" to it. In this case, 1 is 2**23 because we are going to divide this whole thing by 2**23
mantissa=(numerator/2**23) #divide the whole thing by 2**23



totalnumerator=(numerator)*2**e # now we are multiplying by 2** exponent to get the whole number as an exponent
print("exponent")
print(e)
print("mantissa numerator")
print(numerator)
print("total numerator")
print(totalnumerator)
print("denominator")
print(2**23)
print('the fraction at 32 bit precision is')
y32=np.float32(totalnumerator/2**23) #im going to call y the calculated value and x the original value
print(y32)
print('the fraction at 64 bit precision is')
y64=np.float64(totalnumerator/2**23) 
print(y64)
print('the fraction at 128 bit precision is')
y128=np.float128(totalnumerator/2**23)
print(y128)







"""

def get_mant(bits):

    m=np.float64(1) # the sum starts at m=1
    for i in range(len(bits)):
        
        m=m+np.float64((bits[i])*2.0**-(i+1))# make sure its float 64

    return(m)



m=get_mant(frac)
print("mantissa is in decimal form is " + str(mantissa))
print("exponent in decimal form is:" + str(e))
value128=np.float128((2**e)*mantissa)
value64=np.float64((2**e)*mantissa)
diff=np.float128(value128)-np.float128(x)
print("the value with float 64 precision is: " + str(value64))
print("the value with 128 bit precision is : " + str(value128))


"""

