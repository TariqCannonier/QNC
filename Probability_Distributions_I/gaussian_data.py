import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

mu = 8
sigma = 1 #10
N = int(1e4)

# Get samples
samples = np.random.normal( mu, sigma, N )

# Plot theoretical pdf in red
nbins = 100
counts, edges = np.histogram( samples, bins=nbins )
x_axis = ( edges[ 1: ] + edges[ :-1] ) / 2
n_pdf = np.divide( counts, np.trapz( counts, x_axis ) )
fig, ax = plt.subplots()
ax.plot(x_axis, st.norm.pdf( x_axis, mu, sigma ), 'r-', linewidth=2 )

# Plot CD11c using 9 bins
nbins = 9
counts, edges = np.histogram( samples, bins=nbins )
x_axis = ( edges[ 1: ] + edges[ :-1] ) / 2
n_pdf = np.divide( counts, np.trapz( counts, x_axis ) )
ax.bar( x_axis, n_pdf, color='mediumturquoise' )

# Plot Igf1
sigma = 1
samples = np.random.normal( mu, sigma, N )
nbins = 9
counts, edges = np.histogram( samples, bins=nbins )
x_axis = ( edges[ 1: ] + edges[ :-1] ) / 2
n_pdf = np.divide( counts, np.trapz( counts, x_axis ) )
idx = [ (nbins // 2 ) - 1, nbins // 2, (nbins // 2 ) + 1 ]
n_pdf[ idx ] *= .6
ax.bar( x_axis, n_pdf, color='khaki' )

plt.title(f"Gaussian pdf, mu={mu:.2f}, sigma={sigma:.2f}")
plt.xlabel('Value')
plt.ylabel('Probability')
plt.legend([ 'Simulated', 'CD11c', 'Igf1'])

xticklabels = [ '0.6 - 0.65',
                    '0.65 - 0.7',
                    '0.7 - 0.75',
                    '0.75 - 0.8',
                    '0.8 - 0.85',
                    '0.85 - 0.9',
                    '0.9 - 0.95',
                    '0.95 - 1.0',
                    '1.0 - 1.5' ]
ax.set_xticks(x_axis, labels=xticklabels, rotation=45)

plt.show()
