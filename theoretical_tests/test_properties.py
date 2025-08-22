from sympy import symbols, simplify, expand

s, t, b1, b2, b, k = symbols('s, t, b1, b2, b, k')

beta_left=(b1*s**2*(3*t-2*s)+b2*(t-s)**2*(t+2*s))/t**3 # t>=s
beta_right=(b1*(s-t)**2*(3-2*s-t)+b2*(1-s)**2*(1+2*s-3*t))/(1-t)**3 # t<=s

# Case t=s
beta_left.subs({t: s}) # = b1
beta_right.subs({t: s}) # = b2

# Case b1=b2=b
simplify(expand(beta_left.subs({b1: b, b2: b}))) # =b
simplify(expand(beta_right.subs({b1: b, b2: b}))) # =b

# Weights functions W's

W_left = s**2*(3*t-2*s)/t**3
W_right = (s-t)**2*(3-2*s-t)/(1-t)**3

beta_left_w = b1*W_left+b2*(1-W_left)
simplify(expand(beta_left_w)-expand(beta_left))# =0. Proof exp. w_left

beta_right_w = b1*W_right+b2*(1-W_right)
simplify(expand(beta_right_w)-expand(beta_right))# =0. Proof exp. w_right

# Limits
W_left.subs({s:t})
W_left.subs({s:0})
W_right.subs({s:t})
W_right.subs({s:1})

# Symmetry on W's.
simplify(expand(W_left + W_right.subs({t:1-t, s:1-s})))

#Temporal invariance: beta_left(t,s)=beta_left(tk,sk)
simplify(expand(beta_left.subs({t:k*t, s:k*s}))-expand(beta_left))

#Temporal invariance: beta_right(t,s)=beta_right(k(1-t)+1,k(1-s)+1)
simplify(expand(beta_right.subs({t:k*(1-t)+1, s:k*(1-s)+1}))-expand(beta_right))