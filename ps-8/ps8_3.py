import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(10, 6)) 
filepath=os.getcwd()
filepath='/Users/Krishna/Documents/repositories/phys-ga2000/ps-8'

dow=np.loadtxt(f'{filepath}/dow.csv')
plt.plot(dow)
plt.xlabel('Time (days)')
plt.ylabel('DJI (USD)')
plt.savefig(f'{filepath}/figs/dow.png')
dow_fft=np.fft.rfft(dow)


tenPercent=int( (len(dow_fft-1))/10 ) # find the index of the first 10% of the elements and convert it to an integer

dow_10_fft=dow_2_fft=dow_fft
dow_10_fft[tenPercent:]=0 # for all indexes above the first 10% of indexes, set the elments to zero

print(len(dow_fft))
print(dow_10_fft)
dow_10=np.fft.irfft(dow_10_fft)

#same procedure
twoPercent=int( (len(dow_fft))/50 )

dow_2_fft[twoPercent:]=0

dow_2=np.fft.irfft(dow_2_fft)

plt.plot(dow, alpha=.6, c='green', label='Unfiltered Dow') 

plt.plot(dow_10,alpha=0.7, c='purple',label='First 10% Harmonics')

plt.plot(dow_2,alpha=0.8, c='pink', label='First 2% Harmonics')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('DJI (USD)')
plt.savefig(f'{filepath}/figs/dow_10.png')
plt.show()





