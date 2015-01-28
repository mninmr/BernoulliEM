'''
BernEM.py

This is an implementation of EM algorithm on binary dataset
with Bernoulli likelihood. 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SEED = 483982
PRNG = np.random.RandomState(SEED)

## NUMBER OF CLUSTERS ##
K = 2
####

def generateSample(N, D, pi, mu):
	sample = np.ndarray([N, D])
	for ii in range(0, N):
		zn = np.random.multinomial(1, pi, size=1)[0]
		ind = _indices(zn, lambda x: x==1)[0]
		sample[ii, :] = np.random.binomial(1, mu[ind])	
	return sample

def _indices(a, func):
	return [i for (i, val) in enumerate(a) if func(val)]

def em(observations, cont_tol, iterations):
	[N, D] = observations.shape
	iteration = 1
	delta_change = 9999

	# Init Model
	pi = [.5, .5]
	mu = [[.5, .5, .5, .5], [.9, .1, .1, .9]]
	r = np.zeros([N, K]) # soft assignment
	weight = np.zeros(K)

	# Main loop
	while iteration <= iterations:
		# E step
		for ii in range(0, N):
			observation = observations[ii]
			for kk in range(0, K):
				weight[kk] = pi[kk]
				for jj in range(0, D):
					weight[kk] *= mu[kk][jj]**observation[jj] * (1-mu[kk][jj])**(1-observation[jj])
			r[ii, :] = weight / sum(weight)

		# M step
		nk = [sum(r[:, ii]) for ii in range(K)]

		new_mu = np.zeros([K, D])
		for kk in range(0, K):
			mean = np.zeros(D)
			for ii in range(0, N):
				mean += r[ii, kk]*observations[ii]
			new_mu[kk] = mean / nk[kk]
		pi = nk / sum(nk) 

		delta_change = sum(sum(abs(new_mu-mu)))
		if delta_change < cont_tol:
			break
		else:
			mu = new_mu
			iteration += 1

	return [mu, pi, iteration]


def main():
	num_feat = 4
	num_sample = 1000
	cont_tol = 1e-6
	max_iter = 1000

	# GLOBAL PARAMS
	pi = [.2, .8] # probablity of each cluster
	mu = [[.5, .5, .5, .5], [.9, .9, .1, .1]] # probabilities of 1 for each cluster

	# Generate observations
	observations = generateSample(num_sample, num_feat, pi, mu)

	# Run EM
	[mu_est, pi_est, iterations] = em(observations, cont_tol, max_iter)


	print 'mu', mu_est
	print 'pi', pi_est
	print 'iter', iterations


if __name__ == '__main__':
	import time
	start_time = time.time()
	main()
	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))





