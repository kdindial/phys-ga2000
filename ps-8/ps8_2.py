import numpy as np
import matplotlib.pyplot as plt
import os

filepath=os.getcwd()
filepath='/Users/Krishna/Documents/repositories/phys-ga2000/ps-8'

piano=np.loadtxt(f'{filepath}/piano.csv')
trump=np.loadtxt(f'{filepath}/trumpet.csv')

sample_rate=44100 #the frequency of timesteps: 1/dt
piano_len=len(piano)/sample_rate #the total legnth in time of the piano sample
trump_len= len(trump)/sample_rate # the total legnth in time of the trumpet sample


piano_f=np.fft.rfft(piano)
trump_f= np.fft.rfft(trump)

N=10000 # first 10k coeficcients

piano_frequencies= np.arange(N)/(2*piano_len)
trump_frequencies= np.arange(N)/(2*trump_len )


plt.plot(np.arange(len(piano))/sample_rate, piano)
plt.xlabel('time (s)')
plt.ylabel('intensity')
plt.title('Piano Waveform')

plt.savefig(f'{filepath}/figs/piano.png')

plt.clf()

plt.plot(np.arange(len(trump))/sample_rate, trump )
plt.title('Trumpet Waveform')
plt.xlabel('time (s)')
plt.ylabel('intensity')
plt.savefig(f'{filepath}/figs/trumpet.png')



plt.clf()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax[0].plot(piano_frequencies, np.abs(piano_f[:N]), label="Piano  Spectrum")
ax[1].plot(trump_frequencies, np.abs(trump_f[:N]), label= 'Trumpet Spectrum')
ax[1].set_title('Trumpet')
ax[0].set_title('Piano')
ax[1].set_xlabel('Frequencies (Hz)')
ax[0].set_xlabel('Frequencies (Hz)')
fig.text(0.04, 0.5, '|Fourier Coefficients|', va='center', rotation='vertical')
plt.subplots_adjust(hspace=0.4) 


piano_fundamental_idx= np.argmax(piano_f)
trumpet_fundamental_idx=np.argmax(trump_f[:2000])
plt.savefig(f'{filepath}/figs/problem2.png')

#Usually the not
piano_fuandemntal=piano_fundamental_idx* (sample_rate/(2*len(trump) ))
trumpet_fuandmental=trumpet_fundamental_idx* (sample_rate/(2*len(trump) ))

print(f"The trumpet is playing a note at {trumpet_fuandmental}Hz")
print(f"The piano is playing a note at {piano_fuandemntal} Hz")


