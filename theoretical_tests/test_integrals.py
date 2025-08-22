from sympy import symbols, integrate, Matrix
import numpy as np
import math

# (A) integral (3*x+c)**4/(1-x)**12 de 0 a k.
k=0.1
c = -0.5
V_left = np.array((((1-k)**(-7)-1)/7, ((1-k)**(-8)-1)/8, ((1-k)**(-9)-1)/9, ((1-k)**(-10)-1)/10, ((1-k)**(-11)-1)/11)).reshape(-1,1)
A = np.array(((3+c)**0, -(3+c)**1, (3+c)**2, -(3+c)**3, (3+c)**4))
C = np.array((81, 108, 54, 12, 1))
(C*A@V_left)[0]

x = symbols('x')
integral = integrate((3*x+c)**4/(1-x)**12,x)
integral.subs(x,k)-integral.subs(x,0)

# (B) integral (3*x+c)**4/x**12 de k a 1.
k=0.5
c = -0.5
V_right = np.array(((k**(-7)-1)/7, (k**(-8)-1)/8, (k**(-9)-1)/9, (k**(-10)-1)/10, (k**(-11)-1)/11)).reshape(-1,1)
A = np.array((c**0, c**1, c**2, c**3, c**4))
C = np.array((81, 108, 54, 12, 1))
(C*A@V_right)[0]

x = symbols('x')
integral = integrate((3*x+c)**4/x**12,x)
integral.subs(x,1)-integral.subs(x,k)

# (C) integral (3*x+c1)**2*(3*x+c2)**2/(1-x)**12 de 0 a k.
k=0.5
c1 = -0.65454
c2 = -0.78955
V_left = np.array((((1-k)**(-7)-1)/7, ((1-k)**(-8)-1)/8, ((1-k)**(-9)-1)/9, ((1-k)**(-10)-1)/10, ((1-k)**(-11)-1)/11)).reshape(-1,1)
A = np.array([1, -(c1+c2)/2-3, 3*(c1+c2)+(c1**2+c2**2)/6+2*c1*c2/3+9,
-3*(c1**2+c2**2)/2- c1*c2*(c1+c2)/2-6*c1*c2-27*(c1+c2)/2-27,
c1**2*c2**2+9*(c1**2+c2**2)+6*c1*c2*(c1 + c2)+36*c1*c2+54*(c1+c2)+81])
C = np.array([81, 108, 54, 12, 1])
((C*A)@V_left)[0]

x = symbols('x')
integral = integrate((3*x+c1)**2*(3*x+c2)**2/(1-x)**12,x)
integral.subs(x,k)-integral.subs(x,0)

# (D) integral (3*x+c1)**2*(3*x+c2)**2/x**12 de k a 1.
k=0.5
c1 = 3
c2 = 3.5
V_right = np.array(((k**(-7)-1)/7, (k**(-8)-1)/8, (k**(-9)-1)/9, (k**(-10)-1)/10, (k**(-11)-1)/11)).reshape(-1,1)
A = np.array((1, (c1+c2)/2, (c1**2+4*c1*c2+c2**2)/6, c1*c2*(c1+c2)/2, c1**2*c2**2))
C = np.array([81, 108, 54, 12, 1])
(C*A@V_right)[0]

x = symbols('x')
integral = integrate((3*x+c1)**2*(3*x+c2)**2/x**12,x)
integral.subs(x,1)-integral.subs(x,k)

#(E) integral (3*x+c1)**2*(3*x+c2)**2/(x**6*(1-x)**6) de k a l
k=0.2
l=0.3
c1 = -0.5
c2 = -1.6546
M = np.array(
[[math.comb(6,m) for m in range(11)],
[math.comb(7,m) for m in range(11)],
[math.comb(8,m) for m in range(11)],
[math.comb(9,m) for m in range(11)],
[math.comb(10,m) for m in range(11)]
])
f1 = lambda m, a, b: 1/(m-5)*(a**(m-5)-b**(m-5))
f2 = lambda a, b: math.log(a)-math.log(b)
V_midd = np.array([f1(m,(1-k)/k,(1-l)/l) if m != 5 else f2((1-k)/k,(1-l)/l) for m in range(11)]).reshape(-1,1)
A = np.array((1, (c1+c2)/2, (c1**2+4*c1*c2+c2**2)/6, c1*c2*(c1+c2)/2, c1**2*c2**2))
C = np.array([81, 108, 54, 12, 1])
((C*A)@(M@V_midd))[0]

x = symbols('x')
integral = integrate((3*x+c1)**2*(3*x+c2)**2/(x**6*(1-x)**6),x)
integral.subs(x,l)-integral.subs(x,k)
