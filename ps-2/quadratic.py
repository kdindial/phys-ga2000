import numpy as np

def quadratic(a,b,c):
    q=np.sqrt(b**2-4*a*c)

    if b>0:

        #if b is positive, we want to add q to get something small
        x1=np.float32((-b+q)/(2*a))
        x2=np.float32((2*c)/(-b+q))

        return(x1,x2)
    else:

    #if b is negative, then we want to subtract q to get something small
        x1=np.float32((-b-q)/(2*a))
        x2=np.float32((2*c)/(-b-q))
        return(x1,x2)




x1,x2=quadratic(0.001,1000,0.001)
print(np.float32(x1))
print(x2)





