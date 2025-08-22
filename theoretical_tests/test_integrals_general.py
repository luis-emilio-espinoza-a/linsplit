from scipy.integrate import quad
import numpy as np
import math

def int1_np(a1, a2, b1, b2, n, m, l):
    #\int_{l}^{1}(a_{1}x+b_{1})^{n}(a_{2}x+b_{2})^{n}/x^{n+m+1}\,dx
    C = np.zeros((2*n+1,1))
    for i in range(n+1):
        for j in range(n+1):
            C[i+j][0] += math.comb(n,i)*math.comb(n,j)*a1**(n-i)*a2**(n-j)*b1**i*b2**j
    L = np.zeros((2*n+1,1))
    for h in range(2*n+1):
        L[h][0] = (l**(-m+n-h)-1)/(m-n+h)
    return (C.T@L)[0][0]

def int2_np(a1, a2, b1, b2, n, m, k):
    #\int_{0}^{k}(a_{1}x+b_{1})^{n}(a_{2}x+b_{2})^{n}/(1-x)^{n+m+1}\,dx
    a1p, a2p = -a1, -a2
    b1p, b2p = a1+b1, a2+b2
    C = np.zeros((2*n+1,1))
    for i in range(n+1):
        for j in range(n+1):
            C[i+j][0] += math.comb(n,i)*math.comb(n,j)*a1p**(n-i)*a2p**(n-j)*b1p**i*b2p**j
    K = np.zeros((2*n+1,1))
    for h in range(2*n+1):
        K[h][0] = ((1-k)**(-m+n-h)-1)/(m-n+h)
    return (C.T@K)[0][0]

def int3_np(a1, a2, b1, b2, n, m, k, l):
    #\int_{k}^{l}(a_{1}x+b_{1})^{n}(a_{2}x+b_{2})^{n}/(x^{m}(1-x)^{m})\,dx
    kp, lp  = (1-k)/k, (1-l)/l 
    fun = lambda h_q: (kp**(m-2*n-1+h_q)/(m-2*n-1+h_q))-(lp**(m-2*n-1+h_q)/(m-2*n-1+h_q)) if m-2*n-2+h_q != -1 else np.log(kp/lp)
    C = np.zeros((2*n+1,1))
    for i in range(n+1):
        for j in range(n+1):
            C[i+j][0] += math.comb(n,i)*math.comb(n,j)*a1**(n-i)*a2**(n-j)*b1**i*b2**j
    M = np.zeros((2*n+1,2*m-1))
    for h in range(2*n+1):
        for q in range(2*m-1):
            if 2*(m-n-1)+h>=q:
                M[h][q] = math.comb(2*(m-n-1)+h, q)

    V = np.zeros((2*m-1,1))
    for q in range(2*m-1):
        V[q][0]= fun(-2*(m-n-1)+q)
    return (C.T@M@V)[0][0]

a1, a2 = 1, 2
b1, b2 = -3.12323, 4.465
n, m = 4, 9
k, l = 0.4, 0.65

int1_np(a1, a2, b1, b2, n, m, l)
quad(lambda x: (a1*x+b1)**n*(a2*x+b2)**n/x**(n+m+1), l, 1, epsabs=1e-12, epsrel=1e-12)[0]

int2_np(a1, a2, b1, b2, n, m, k)
quad(lambda x: (a1*x+b1)**n*(a2*x+b2)**n/(1-x)**(n+m+1), 0, k, epsabs=1e-12, epsrel=1e-12)[0]

int3_np(a1, a2, b1, b2, n, m, k, l)
quad(lambda x: (a1*x+b1)**n*(a2*x+b2)**n/(x**m*(1-x)**m), k, l, epsabs=1e-12, epsrel=1e-12)[0]
