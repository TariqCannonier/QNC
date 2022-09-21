import numpy as np
import random as rnd
import collections
import matplotlib.pyplot as plt
import scipy.stats as st
import math
from scipy.stats import binom
from scipy.misc import factorial
from decimal import Decimal

# P(x=successes) = (n_choose_k) * p^k * (1-p)^(n-k)
"""
Exercise 1
"""
p = 0.2
quanta = np.arange(0,11)
P = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None,
    6:None, 7:None, 8:None, 9:None, 10:None}
n=10

for k in quanta:
    P[k] = (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*(0.2**k)*((1-0.2)**(n-k))
print('Exercise1:\n')
print(P)


"""
Exercise 2
"""
release_probability = np.arange( 0, 1.1, 0.1 )
k = 8
P = []
for r in release_probability:
    P.append((math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))*(r**k)*((1-r)**(n-k)))
print("\nExercise2:")
print(P)


"""
Exercise 3

"""
prob = dict(p5=[],p8=[])
n = 14
for r in release_probability:
    prob['p5'].append((math.factorial(n)/(math.factorial(5)*math.factorial(n-5)))*(r**5)*((1-r)**(n-5)))
    prob['p8'].append((math.factorial(n)/(math.factorial(8)*math.factorial(n-8)))*(r**8)*((1-r)**(n-8)))

likelihood = np.multiply(prob['p5'],prob['p8'])
log_likelihood = np.log(prob['p5']) + np.log(prob['p8'])

print("\nExercise3:")
print(prob)
print('likelihood: ')
print(likelihood)
print('log likelihood: ')
print(log_likelihood)
print("")


"""
Exercise 4
"""
dr = 0.01
n=100
release_probability = np.arange( 0, 1+dr, dr )

# Is there a missing value in the dataset???
measured_release_count = [ 0, 0, 3, 0, 10, 19, 26, 16, 16, 5, 5, 0, 0, 0, 0 ]

measured_release = np.arange(15)
P = np.zeros([len(release_probability), len(measured_release_count)])
for idx1, p in enumerate(release_probability):
    for idx2 in measured_release:
        # import pdb; pdb.set_trace()
        binom_coeff = factorial( n ) / ( factorial( measured_release_count[ idx2 ] ) * factorial( n - measured_release_count[ idx2 ] ) )
        P[ idx1, idx2 ] = binom_coeff * p**measured_release_count[ idx2 ] * (1-p)**(n-measured_release_count[idx2])

loglikelihood = np.sum(np.log(P), axis=1)
maxLikelihood = release_probability[np.argmax(loglikelihood)]

print("\nExercise4:\nThe most likely value of p is: " % maxLikelihood)

"""
Exercise 5
"""
p = 0.3
trials = 7
n=14
binom_coeff = factorial( n ) / ( factorial( trials ) * factorial( n - trials ) )
P = binom_coeff * p**trials * (1-p)**(n-trials)


print("\nExercise 5:\nBecause we observed a change in release probability with the change in temperature, the null hypothesis is not supported.  Temperature had an effect on the release probability. P=%2.2f" % P)
