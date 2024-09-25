import numpy as np
import timeit
import matplotlib.pyplot as plt

def matrixMult( A: np.array, B: np.array):

    #check if the cols of A are equal to the rows of B
    if np.shape(A)[1]==np.shape(B)[0]:
        N=np.shape(A)[1]
        #create an NxN array 
        C= np.zeros([N,N], float)
        for i in range (N):
            for j in range(N):
                for k in range(N):
                    C[i,j]+=A[i,k]*B[k,j]

        return(C)
    else:
        raise Exception('cols of A do not equal rows of B') 


#Make an array of N will store the N values that we are going to test and plot. where NXN is the dim of the matrices we will multiply
N = np.linspace(10,110,21).astype(int) 


As,Bs = [],[] #make two lists to store our As and Bs
for n in range(len(N)): 
    As.append(np.random.rand(N[n], N[n])) 
    Bs.append(np.random.rand(N[n], N[n])) 


"""

I told chat gpt:
"I have a function that takes two arguments. I want to use timeit but vary what the argument is

chatgpt told me: 

To use timeit for a function with varying arguments, you can define the function with the arguments you want, and then pass the function call within a lambda to timeit.timeit. This allows you to vary the arguments each time you run timeit.

Here’s an example:

# Example function
def my_function(x, y):
    return x * y

# Varying arguments
x_values = [1, 2, 3, 4]
y_values = [5, 6, 7, 8]

# Timeit for different argument pairs
for x, y in zip(x_values, y_values):
    t = timeit.timeit(lambda: my_function(x, y), number=10000)
    print(f'Time for my_function({x}, {y}): {t}')

"""
t1=[]
for A, B in zip(As,Bs):
    t1.append(timeit.timeit(lambda:matrixMult(A,B), number=1))

M=np.linspace(10,1100,101).astype(int) 
As,Bs = [],[] #make two lists to store our As and Bs
for n in range(len(M)): 
    As.append(np.random.rand(M[n], M[n])) 
    Bs.append(np.random.rand(M[n], M[n])) 
t2=[]

for A, B in zip(As,Bs):
    t2.append(timeit.timeit(lambda:np.dot(A,B), number=1))
#make t1 and t2 numpy arrays so that i can multiply do math with them

"""
better way to do it. Plot log(t), log(N), then take the slope of that curve and that is the log scale. 

"""
exp1 = np.polyfit(np.log(N),np.log(t1),1)[0]
a=np.polyfit(np.log(N),np.log(t1),1)[1]
print('the slope of log(N) vs log(t) for our for loop method is:  '+ str(exp1))
exp2 = np.polyfit(np.log(M),np.log(t2),1)[0]
print('the slope  of log(N) vs log(t) for of the dot method is: '+ str(exp2))


t1=np.array(t1)
t2=np.array(t2)
N=np.array(N)
fig, ax = plt.subplots(2)




ax[0].scatter(N,t1*10**3, c='purple')
#ax[0].plot(N,np.exp(a+exp1*N), c='black', label='$linear fit of log(N)$')
#ax[0].legend()
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel("Size of Array")
ax[0].set_ylabel("computation time (ms)")
ax[0].set_title("Matrix Multiplication Using a For Loop")
ax[1].scatter(M,t2*10**3)
ax[1].set_xlabel("Size of Array")
ax[1].set_ylabel("computation time (ms)")
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_title("Matrix Multiplication Using np.dot")
plt.subplots_adjust(hspace=0.5)
plt.savefig('ps-3-1.png')

plt.show()