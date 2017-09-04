#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:22:45 2017

@author: ivan
"""

import numpy as np

class MultivGauss:
    ''' A little bit about MultivGauss class'''
    def __init__(self, x):
        self.x = x
        self.n = x.shape[1] # dimension
        self.m = x.shape[0] # n samples
        self.mu = None  # np.mean(x, axis=0)
        self.sigma = None # 1/self.n * sum
        self.values = np.empty(self.m)

    def dot_product_transposed(self, x):
        """
        en teoría deberíamos un tile(x,len(x)) de x para multiplicarlo por x_t,
        pero lo auto ajusta.
        
        x_t[1] * x
        x_t[2] * x
        x_t[3] * x
        
        donde x = (x[1], x[2], x[3], ...)
        """
        x_t = x[:,np.newaxis]
        return x_t * x
    
    def compute_mu(self):
        self.mu = np.mean(self.x, axis=0)
    
    def compute_sigma(self):
        A = self.x - self.mu
        B = np.empty((self.m,self.n,self.n))
        for i in range(self.m):
            B[i] = self.dot_product_transposed(A[i])
        self.sigma = 1/self.m * np.sum(B,axis=0)

    def value(self, point):
        cte = (2 * np.pi)**(self.n * 0.5) * np.abs(np.sqrt(np.linalg.det(self.sigma)))
        A = point - self.mu
        return (1/cte) * np.exp(-0.5 * np.dot(A,np.dot(A,np.linalg.inv(self.sigma))))
        
    def compute_values(self):
        for i in range(self.m):
            point = self.x[i,:]
            self.values[i] = self.value(point)
            
    def run(self):
        self.compute_mu()
        self.compute_sigma()
        self.compute_values()
        
"""
def mult_normal(x, MU, SIGMA):
    cte = (2 * np.pi)**(N/2) * np.abs(np.sqrt(np.linalg.det(SIGMA)))
    A = x - MU
    return (1/cte) * np.exp(-0.5 * np.dot(A,np.dot(A,np.linalg.inv(SIGMA))))


def dot_product_transposed(x):
    b = np.tile(x,(2,1))
    a = x[:,np.newaxis]
    return a*b

def calcular_params(sample):
    mu = np.mean(sample, axis=0)
    A = sample - mu
    B = np.empty((N_SAMPLES,2,2))
    for i in range(N_SAMPLES):
        B[i] = dot_product_transposed(A[i])
    sigma = 1/N_SAMPLES * np.sum(B,axis=0)
    return mu, sigma
"""

if __name__ == '__main__':
    
    N = 2
    N_SAMPLES = 100
#    MU = np.array([0.6, 0.75, 2])
#    SIGMA = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0.3]])

    MU = np.array([1,2])
    SIGMA = np.array([[1, 0.5], [0.5, 1]])
    
    sample = np.random.multivariate_normal(MU, SIGMA, N_SAMPLES)
    
    
    mu, sigma = calcular_params(sample)
    
    mv = MultivGauss(sample)
    mv.compute_mu()
    mv.compute_sigma()
    
    #print(calcular_params(sample)[1])
    print(mu)
    print(sigma)
    print(mv.mu)
    print(mv.sigma)
    
    print(mult_normal(sample[2], mu, sigma))
    print(mv.value(sample[2]))