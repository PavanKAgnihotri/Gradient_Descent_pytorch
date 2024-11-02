#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:22:08 2024

@author: pavan
"""


import torch

def R(z): 
    return torch.sum(z * z * z * z) 

def grad_R(z):
    return 4 * (z * z * z)

# Gradient Descent
z = torch.FloatTensor([-3, 2])
print('Iteration 0: z =', z)
alpha = 1
tol = 1e-6
count = 0
while True:
    z_next = z - alpha * grad_R(z)
    count += 1
    print('Iteration', count,': z =', z_next)
    if torch.norm(z - z_next) < tol:
        break
    z = z_next
