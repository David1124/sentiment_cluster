#!/usr/bing/env python
#coding=utf8

from __future__ import with_statement
import cPickle as pickle
from matplotlib import pyplot
from numpy import zeros, array, tile
from scipy.linalg import norm
import numpy.matlib as ml
import random

def kmeans(X, k, observer=None, threshold=1e-15, maxiter=300):
	N = len(X)
	labels = zeros(N, dtype=int)
	centers = array(random.sample(X, k))
	iter = 0

	def calc_J():
		sum = 0
		for i in xrange(N):
			sum += norm(X[i]-centers[labels[i]])
		return sum

	def distmat(X, Y):
		n = len(X)
		m = len(Y)
		xx = ml.sum(X*X, axis=1)
		yy = ml.sum(Y*Y, axis=1)
		xy = ml.dot(X, Y.T)
		
		return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy
	
	Jprev = calc_J()
	while True:
		#notify the observer
		if observer is not None:
			observer(iter, labels, centers)

		dist = distmat(X, centers)
		labels = dist.argmin(axis=1)
		for j in range(k):
			idx_j = (labels == j).nonzero()
			distj = distmat(X[idx_j], X[idx_j])
			distsum = ml.sum(distj, axis=1)
			icenter = distsum.argmin()
			centers[j] = X[idx_j[0][icenter]]

		J = calc_J()
		iter += 1
		
		if Jprev-J < threshold:
			break
		Jprev = J
		if iter >= maxiter:
			break
	
	if observer is not None:
		observer(iter, labels, centers)

if __name__ == "__main__":
	with open("cluster.pkl") as inf:
		samples = pickle.load(inf)
	print samples
	N = 0
	for smp in samples:
		N += len(smp[0])
	X = zeros((N, 2))
	idxfrm = 0
	for i in range(len(samples)):
		idxto = idxfrm + len(samples[i][0])
		X[idxfrm:idxto, 0] = samples[i][0]
		X[idxfrm:idxto, 1] = samples[i][1]
		idxfrm = idxto
	
	def observer(iter, labels, centers):
		print "iter %d." % iter
		colors = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		pyplot.plot(hold=False)
		pyplot.hold(True)

		data_colors = [colors[lbl] for lbl in labels]
		pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
		pyplot.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)

		pyplot.savefig("kmediods/iter_%02d.png" % iter, format="png")
	
	kmeans(X, 3, observer=observer)
