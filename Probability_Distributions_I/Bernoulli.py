import random as rnd
from scipy.stats import bernoulli

# Generate a random value less than p
p = 0.7
single_outcome_method_1 = rnd.random() < p
print(f"outcome using rand = {single_outcome_method_1}")

# Generate a rnadom value less than p using bernoulli function
single_outcome_method_2 = bernoulli.rvs(p, size=1)
print(f"outcome using binornd = {bool(single_outcome_method_2)}")

# Lots of Bernoulli trials to check convergence
p = 0.7
N = int(1E4)
print(N)
outcomes = bernoulli.rvs(p, size=N)

# Print out of Bernoulli trial results
print(f"{(outcomes == False).sum()} zeros, {(outcomes == True).sum()} ones, simulated p = {(outcomes == True).sum()/outcomes.size}, empirical p = {p}")
