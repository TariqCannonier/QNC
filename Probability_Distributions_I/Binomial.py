import numpy as np
import random as rnd
import collections
import matplotlib.pyplot as plt
import time
import scipy.stats as st

from scipy.stats import bernoulli, binom, poisson, chi2
from IPython.display import clear_output
from operator import itemgetter
from statsmodels.stats import proportion

from numpy import matlib



# Paramters for n and p
p = 0.7
n = 1000

# Generate random picks
num_successes = binom.rvs(n,p) # uses the random variates method (rvs) of binom
print(f"{num_successes} successes out of {n} trials, simulated p = {p:.4f}, empirical p = {num_successes/n:.4f}")


# Simulate many different picks
p = 0.7
n = 10
num_experiments = 1000
outcomes = binom.rvs(n, p, size=num_experiments)

# Plot histogram of all possible outcomes
counts = collections.Counter(outcomes)
plt.subplot(211)
plt.bar(counts.keys(), counts.values())
plt.xlim([-1, n+1])
plt.title(f"Binomial distribution, n={n}, p={p:.2f}, {num_experiments} simulations")
plt.ylabel('Count')

# Show normalized version that is a pdf
normalized_counts = { k: v / total for total in (sum(counts.values()),) for k, v in counts.items()}
plt.subplot( 212 )
plt.bar( normalized_counts.keys(), normalized_counts.values())
plt.xlim( [ -1, n+1 ] )
plt.xlabel( f"Probability of success in {n} tries" )
plt.ylabel( "Probability" )
plt.subplots_adjust( hspace=0.3 )
plt.show()

# Animate
n = 10
xs = range( 0, n )
f = plt.figure()

for p in np.arange( 0, 1, 0.1 ):
    for N in np.round( np.logspace( 1, 5, 10 ) ):
        # Get true binomial pdf
        Y = binom.pmf( xs, n, p )

        # Get random picks and make histograms/normalize
        counts = collections.Counter(binom.rvs( n, p, size=int(N) ) )
        normalized_counts = { k: v / total for total in ( sum( counts.values() ), ) for k, v in counts.items() }

        # Show both
        plt.bar( normalized_counts.keys(), normalized_counts.values() )
        plt.plot( xs, Y, 'ro-', linewidth=2, markersize=10 )

        # Labels, etc
        plt.title(f"p={p:.1f}, n={n}, N={N:.2f}")
        plt.xlabel(f"Number of successes in {n} tries")
        plt.ylabel("Probability")
        plt.axis( [ -1, n+1, 0, 0.45 ] )
        plt.legend( ("Theoretical", "Simulated" ) )
        plt.show(block=False)

        # Wait
        time.sleep(0.1)
        plt.close('all')
        #Clear for next Plot
        clear_output(wait=False)


# Comulative distribution
p = 0.7
n = 10
num_experiments = int( 1e3 )
outcomes = binom.rvs( n, p, size=num_experiments )

# Make histograms
counts = collections.Counter(outcomes)
keys, values = zip( *sorted( counts.items(), key=itemgetter( 0 ) ) )

# Compute cumulative sum of the counts normalized by total normalized_counts
total = sum( values )
cumulative_ps = [ x/total for x in np.cumsum( values ) ]

# Plot as bar graph
plt.bar( keys, cumulative_ps )

# Compare to real Binomial cumulative distribution
Y = binom.cdf( keys, n, p )
plt.plot( keys, Y, 'ro-', linewidth=2, markersize=10 )

# Label Plot
plt.title( f'Cumulative binomial distribution, p={p:.1f}, n={n}, N={num_experiments}' )
plt.xlabel( f'Number of successes in {n} tries' )
plt.ylabel( 'Cumulative probability' )
plt.legend( ( 'Theoretical', 'Simulated') )
plt.show();
