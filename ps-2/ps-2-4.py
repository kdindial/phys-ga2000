import numpy as np
import matplotlib.pyplot as plt
N=1000
#=np.zeros((N,N)) # a grid representing the complex plane. the first index is the real number and the second index is the imaginary number

"""
I asked chatgpt:

can you write a function that generates an NxN grid representing the complex plane where x is the real axis from -2 to 2 and y is the imaginary axis from -2i to 2i?

here is what it gave me:
"""
def complex_grid(N):
    # Create linearly spaced real and imaginary parts
    real_axis = np.linspace(-2, 2, N)
    imaginary_axis = np.linspace(-2, 2, N) * 1j
    
    # Generate the grid by adding the real and imaginary components
    grid = np.array([[real + imag for real in real_axis] for imag in imaginary_axis])
    
    return grid

"""
ok i could have done that, but i was feeling lazy


then I made a function to test whether or not a given complex number c is in the mandle set
it will return - if its not in the mandelbrot set and 1 if it is in the mandelbrot set
"""


def mandle_test(c , max_it=100):
    z=0
    for i in range (max_it):
        z=z**2+c
        if abs(z) > 2:
            return 0
    return 1

"""
first i wrote this code to transform from the complex grid to a grid of 0 and 1 where 0 corresponds to the c not being in the set and 1 corresponds to c being in the set:


grid=complex_grid(N)
for i in range(N):
    for j in range(N):
        mandel[i,j]=mandle_test(grid[i,j])

then i asked chatgpt to re-write this code without a for loop. below is what chatgpt gave me:


grid = complex_grid(N)
mandel = np.vectorize(mandle_test)(grid)
"""

grid = complex_grid(N)
#i guess np.vectorize alows you to apply the mandle_test function to each element of grid without the need using a loop.
mandel = np.vectorize(mandle_test)(grid)

plt.imshow(mandel, extent=[-2, 2, -2, 2]) #extent forces the x and y axis to go from -2 to 2, otherwise x and y value would correspond to the index of mandel
plt.ylabel('imaginary')
plt.xlabel('real')
plt.title('mandelbrot plot')
plt.show()

plt.savefig('mandelbrotPlot.png')