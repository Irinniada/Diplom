import os
import sys
import numpy as np

def TDMA(a,b,c,f):
    #a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))
    alpha = [0]
    beta = [0]
    n = len(f)

    x = [0]*n

    for i in range(n-1):
        alpha.append(b[i]/(c[i] - a[i]*alpha[i]))

        beta.append((a[i]*beta[i]+f[i])/(c[i] - a[i]*alpha[i]))


    x[n-1] = 0#(f[n-1] - a[n-2]*beta[n-1])/(c[n-1] + a[n-2]*alpha[n-1])

    for i in reversed(range(n-1)):
        x[i] = alpha[i+1]*x[i+1] + beta[i+1]

    return x

