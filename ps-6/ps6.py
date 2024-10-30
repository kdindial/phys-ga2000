import numpy as np
import matplotlib.pyplot as plt 
import astropy.io.fits
import os 
import random

filepath=os.getcwd()
filepath='/Users/Krishna/Documents/repositories/phys-ga2000/ps-6'
hdu_list = astropy.io.fits.open('/Users/Krishna/Documents/repositories/phys-ga2000/ps-6/specgrid.fits')
logwave= hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

wave=np.power(10,logwave)/10


l=3
m=2
fig, ax = plt.subplots(l,m, figsize=(14, 14))

plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing

ax[0,0].set_ylabel(r'Flux $(10^{-17 erg s^{-1} cm^{-2} A^{-1})')

for i in range(l):
    for j in range(m):
        k=random.randint(0,wave.size)
        ax[i,j].plot(wave,flux[k])
        ax[i,j].set_xlabel(r'$\lambda$(nm)')
        ax[i,j].set_ylabel(r'Flux $(10^{-17 erg s^{-1} cm^{-2} A^{-1})')
        ax[i,j].set_title(f'galaxy{k}')


plt.title('Galazy Spectra')

plt.savefig(f'{filepath}/ps6_figs/partA')
plt.clf()

'''
Part B
'''

fluxnorm=np.zeros(flux.shape )
#normalize the data with mean normalization
for i in range (flux[:,0].size):
    fluxnorm[i,:]=flux[i,:]/np.sum(flux[i,:])


'''
Part C
'''

r=np.zeros(flux.shape )
mean_spectra=np.zeros(flux.shape)


for i in range (fluxnorm[:,0].size):
    mean_spectra[i,:]=np.mean(fluxnorm[i,:])
    r[i,:]=fluxnorm[i,:]-mean_spectra[i,:]


plt.subplots_adjust(hspace=.5)
fig, ax = plt.subplots(2,2, figsize=(10, 8))
for i in range(2):
    for j in range(2):
        k=random.randint(0,wave.size)
        ax[i,j].plot(wave,r[k,:])
        ax[i,j].set_xlabel(r'$\lambda$(nm)')
        ax[i,j].set_ylabel('flux')
        ax[i,j].set_title(f'galaxy{k}')

plt.clf()





'''
Part D
'''



C = np.dot(r.T, r) 
eig_val_cov, eig_vec_cov = np.linalg.eig(C)




#sort the eigenvectors
idx_eig_cov = np.argsort(eig_val_cov)[::-1]  # Sort in descending order
eig_cov_sorted =eig_vec_cov[:, idx_eig_cov]





fig, ax=plt.subplots(5,2, figsize=(14, 8))
plt.subplots_adjust(hspace=.5)


for i in range(5):
    ax[i,0].plot(wave,eig_cov_sorted[:, i], label=f'Eigenvector {i + 1}')
ax[0,0].set_title('First 5 Eigenvectors from Diagonalizing the Covariance Matrix')
ax[4,0].set_xlabel('wavelength (nm)')


'''
Part E
'''


u, w, vt = np.linalg.svd(r, full_matrices=False)


eig_vec_svd = vt.T  # Eigenvectors from SVD are stored in `vt.T`

#now sort and normalize v:
idx_eig_svd = np.argsort(w)[::-1]  # Sort in descending order
eig_svd_sorted =eig_vec_svd [:, idx_eig_svd]




for i in range(5):
     ax[i,1].plot(wave, eig_svd_sorted[:, i], label=f'Eigenvector {i + 1}')

ax[0,1].set_title('First 5 Eigenvectors from SVD decomposition of R')
ax[4,1].set_xlabel('wavelength (nm)')
plt.savefig(f'{filepath}/ps6_figs/first_5_eigen_vec')
plt.clf()



'''
Part f
'''
condition_number_r = np.linalg.cond(r)
print(f'Condition number of R Matrix: {condition_number_r}')
condition_number_c = np.linalg.cond(C)

print(f'Condition number of C matrix{condition_number_c}')



"""


Part g

create the approximate spectra by just keeping the first 5 coefficents. each original input spectrum can be expressed as the mean spectru, plus the full set of coefficients ci, 
multiplying the eigenspectra times the original normalization. See what happens if you keep the first 5 coefficients. Dont need to perform any matrix inversion or decopm. just have to rotate the spectra into egienpsectrum basis

"""


def approx_spec(i):
    eigvec = eig_vec_svd[:, 0:i]  # take the first 5 eigenvectors (which have the 5 largest eigenvalues)
    reduced_data = np.dot(eigvec.T, r.T) # project the spectrum into the first 5 eigenvectors
    approx_data = mean_spectra+np.dot(eigvec, reduced_data).T #using the projection, reconstruct the data with the first 5 eigenvectors

    return(approx_data)


fig, ax = plt.subplots(5, figsize=(10, 12))
plt.subplots_adjust(hspace=.8)
plt.title('Approximating the spectra with the first 5 eigenvectors')

for i in range(5):
        ax[i].plot(wave,approx_spec(5)[i,:], color='orange', label='approx spectrum')
        ax[i].set_xlabel(r'$\lambda$(nm)')
        ax[i].set_ylabel('flux')
        ax[i].set_title(f'galaxy{i}')
 
plt.savefig(f'{filepath}/ps6_figs/approxSpectra5')
plt.clf()

'''
Part h
plot c0 vs c1 and c0 vs c2
'''
weights =  w*w

reduced_weights = np.dot(eig_vec_svd.T, r.T).T

plt.figure(figsize=(10, 6))
plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], color='blue', label='c1 vs c0')
plt.scatter(reduced_weights[:, 0], reduced_weights[:, 2], color='green', label='c2 vs c0')
plt.title('Eigenvector Weights (c1, c2 vs c0)')
plt.xlabel('c0')
plt.ylabel('c1 and c2')
plt.legend()

plt.savefig(f'{filepath}/ps6_figs/c0vc1')
plt.clf()
'''
Now try again varying Nc= 1, 2, 20 and calculated the squared rsediuals bebetween the spectra and the aprrox spectra 
'''
Nc=np.arange(1,21,1)

r2=np.zeros(Nc.size)

for i in range(Nc.size):
     residual=(approx_spec(Nc[i])-r)**2
     r2[i]=np.sum(residual)
     
    
plt.figure(figsize=(10, 6))
plt.scatter(Nc, r2)
plt.title('Total Squared Residuals vs Number of Eigenvectors')
plt.xlabel('Number of Eigenvectors')
plt.ylabel('Total Squared Residuals')
plt.savefig(f'{filepath}/ps6_figs/residuals_v_Nc')

