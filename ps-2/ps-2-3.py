import numpy as np

import timeit


def M_loop( L=100 ):

    """
    calculates the madelung constant of a sodium chloride lattice with (2L+1)^3 lattice sites using a for loop


    inputs:
    _____
    L: the legnth our 3d grid extends out from the origin

    outputs:
    _____
    M: float 32, the madelung constant for a cube with sidelength 2L
    
    """

    M=np.float32(0)
    index=np.arange(-L,L+1,1)
    for i in index:
        for j in index:
                for k in index:
                    if i==0 and j==0 and k==0:
                        pass
                        #print("origin")
                    elif (i+j+k)%2==0:
                        M=M+np.float64(1/np.sqrt(i**2+j**2+k**2))
                        #print("++ "+ str(M))
                    else:
                        M=M-np.float64(1/np.sqrt(i**2+j**2+k**2))
                        #print("-- "+ str(M))
    return(M)

#print(" using a for loop and iterating 100 times, M is" + str(M_loop(100)))


def M_no_loop(L=100):
     
    """
    calculates the madelung constant of a sodium chloride lattice with side length 2L


    inputs:
    _____
    L: int 
    the distance our 3d grid extends out from the origin

    outputs:
    _____
    Msum: float 32
    the madelung constant for a cube with sidelength 2L
    
    """
    index=np.arange(-L,L+1)
    i,j,k=np.meshgrid(index,index,index, indexing='ij') 
    even= (i+j+k)%2==0
    odd=(i+j+k)%2!=0
    non_zero= (i!=0) | (j!=0) | (k!=0)

    r=np.sqrt(i**2+j**2+k**2)
    A_even=np.where(non_zero & even, 1/r,0)# make an array where the odd values are filled with 1/r except 0
    A_odd=np.where(non_zero & odd, -1/r,0) # make an array where the odd values are filled with -1/r, everything else is filled with zeros
    A=A_even+A_odd # add them together to get the full array A
    return(np.sum(A))

print("M=" +str(M_no_loop(100)))


t_no_loop = elapsed_time = timeit.timeit('M_no_loop(100)', globals=globals(), number=1)


t_loop = elapsed_time = timeit.timeit('M_loop(100)', globals=globals(), number=1)
print("with no loop " + str(t_no_loop) + "s")
print("with loop " + str(t_loop) + "s")
#print(str(t_loop))