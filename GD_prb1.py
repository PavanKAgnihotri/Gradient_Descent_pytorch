#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:17:52 2024

@author: pavan
"""

import torch

def y_func(x):
    a=torch.pow(x, 3)
    b=torch.pow(x, 2)
    return torch.add(a,b)

def y_derivative(x):
    m=3.0*torch.pow(x,2)
    n=2.0*torch.pow(x,1)
    return torch.add(m,n)

alpha = 0.001
tol = 1e-6

x0= torch.FloatTensor([-2])
print('x0 =', x0, ', y_func =', y_func(x0))
count = 0
while True:
    x_next = x0 - alpha * y_derivative(x0)
    count += 1
    print('Iteration', count)
    print('x=',x_next,' ,y_func=',y_func(x_next))
    if torch.sum(torch.abs(x0-x_next)) < tol:
        break
    x0 = x_next