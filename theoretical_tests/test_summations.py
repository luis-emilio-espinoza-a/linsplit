import numpy as np

def sim_xy(n, s, t):
    a = np.random.rand() # Intercept
    b1 = np.random.normal() # Beta value from 0 to s
    b2 = np.random.normal() # Beta value from s to 1
    x = np.array([(i+1)/n for i in range(n)]) # linspace doesn't work fine.
    y = np.zeros(n)
    y[x<s] = a+b1*x[x<s]
    y[x>=s] = a+x[x>=s]*b2-s*(b2-b1)
    return x, y, a, b1, b2

# CASE 1: s < t

s = 0.3156
t = 0.7753
n = 1_000_000
x, y, a, b1, b2 = sim_xy(n, s, t)

theoretical = {}
empirical = {}

# lenght(]0, s[)
theoretical['lenght(]0,s[)'] = n*s-1
empirical['lenght(]0,s[)'] = x[(x<s)&(x>0)].size

# lenght(]0, t])
theoretical['lenght(]0,t])'] = n*t
empirical['lenght(]0,t])'] = x[(x<=t)&(x>0)].size

# lenght([s, t])
theoretical['lenght([s,t])'] = t*n-s*n+1
empirical['lenght([s,t])'] = x[(x<=t)&(x>=s)].size

# sum[s,t](x)
theoretical['sum[s,t](x)'] = (t+s)*(1+n*(t-s))/2
empirical['sum[s,t](x)'] = x[(x<=t)&(x>=s)].sum()

# sum[s,t](x**2)
theoretical['sum[s,t](x**2)'] = (t*(t*n+1)*(2*t*n+1)-s*(s*n-1)*(2*s*n-1))/(6*n)
empirical['sum[s,t](x**2)'] = (x[(x<=t)&(x>=s)]**2).sum()

# sum]0,t](x)
theoretical['sum]0,t](x)'] = (t*n+1)*t/2
empirical['sum]0,t](x)'] = x[(x<=t)&(x>0)].sum()

# sum]0,t](x**2)
theoretical['sum]0,t](x**2)'] = (t*(t*n+1)*(2*t*n+1))/(6*n)
empirical['sum]0,t](x**2)'] = (x[(x<=t)&(x>0)]**2).sum()

# sum]0,s[(x)
theoretical['sum]0,s[(x)'] = s*(s*n-1)/2
empirical['sum]0,s[(x)'] = x[(x<s)&(x>0)].sum()

# sum]0,s[(x**2)
theoretical['sum]0,s[(x**2)'] = (s*(s*n-1)*(2*s*n-1))/(6*n)
empirical['sum]0,s[(x**2)'] = (x[(x<s)&(x>0)]**2).sum()

# avg ]0,t](x)
theoretical['avg]0,t](x)'] = (t+1/n)/2
empirical['avg]0,t](x)'] = x[(x<=t)&(x>0)].mean()

# denominator(beta]0,t]): 
theoretical['denominator(beta]0,t])'] = t*((t*n)**2-1)/12/n
empirical['denominator(beta]0,t])'] = (x[(x<=t)&(x>0)]**2).sum()-1/(n*t)*((x[(x<=t)&(x>0)]).sum())**2

# sum]0,t](x*y) = 
theoretical['sum]0,t](x*y)'] = (
    a*theoretical['sum]0,t](x)']+
    b1*theoretical['sum]0,s[(x**2)']+
    b2*theoretical['sum[s,t](x**2)']-
    s*(b2-b1)*theoretical['sum[s,t](x)']
)
empirical['sum]0,t](x*y)'] = (x[(x>0)&(x<=t)]*y[(x>0)&(x<=t)]).sum()

# sum]0,t](y)
theoretical['sum]0,t](y)'] = t*n*a-(t*n-s*n+1)*s*(b2-b1)+b1*theoretical['sum]0,s[(x)']+b2*theoretical['sum[s,t](x)']
empirical['sum]0,t](y)'] = y[(x>0)&(x<=t)].sum()

# avg]0,t](y)
theoretical['avg]0,t](y)'] = theoretical['sum]0,t](y)']/(n*t)
empirical['avg]0,t](y)'] = y[(x>0)&(x<=t)].mean()

# beta]0,t]
theoretical['beta]0,t]'] = (theoretical['sum]0,t](x*y)']-t*n*theoretical['avg]0,t](y)']*theoretical['avg]0,t](x)'])/theoretical['denominator(beta]0,t])']
empirical['beta]0,t]'] = np.polyfit(x[(x>0)&(x<=t)], y[(x>0)&(x<=t)], 1)[0]

# CASE 2: s > t

s = 0.7654
t = 0.3159
n = 1_000_000
x, y, a, b1, b2 = sim_xy(n, s, t)

# lenght([t, s])
theoretical['lenght([t,s])'] = n*s-n*t+1
empirical['lenght([t,s])'] = x[(x>=t)&(x<=s)].size

# lenght([t, 1])
theoretical['lenght([t,1])'] = n-t*n+1
empirical['lenght([t,1])'] = x[x>=t].size

# lenght(]s,1])
theoretical['lenght(]s,1])'] = n*(1-s)
empirical['lenght(]s,1])'] = x[x>s].size

# sum_x_t_s
theoretical['sum[t,s](x)'] = (s+t)*(1+n*(s-t))/2
empirical['sum[t,s](x)'] = x[(x>=t)&(x<=s)].sum()

#sum_x2_t_s
theoretical['sum[t,s](x**2)'] = (s*(s*n+1)*(2*s*n+1)-t*(t*n-1)*(2*t*n-1))/(6*n)
empirical['sum[t,s](x**2)'] = (x[(x>=t)&(x<=s)]**2).sum()

#sum_x_t_1
theoretical['sum[t,1](x)'] = (1+t)*(1+n-t*n)/2
empirical['sum[t,1](x)'] = x[x>=t].sum()

#sum_x2_t_1
theoretical['sum[t,1](x**2)'] = ((n+1)*(2*n+1)-t*(t*n-1)*(2*t*n-1))/(6*n)
empirical['sum[t,1](x**2)'] = (x[x>=t]**2).sum()

#sum_x_s_1
theoretical['sum]s,1](x)'] = (n+s*n+1)*(1-s)/2
empirical['sum]s,1](x)'] = x[x>s].sum()

#sum_x2_s_1
theoretical['sum]s,1](x**2)'] = ((n+1)*(2*n+1)-s*(s*n+1)*(2*s*n+1))/(6*n)
empirical['sum]s,1](x**2)'] = (x[x>s]**2).sum()

#avg_x_t_1
theoretical['avg[t,1](x)'] = (1+t)/2
empirical['avg[t,1](x)'] = x[x>=t].mean()

theoretical['denominator(beta[t,1])'] = (1-t)*(n*t-n-1)*(n*t-n-2)/(12*n)
empirical['denominator(beta[t,1])'] = (x[x>=t]**2).sum()-1/(n-t*n+1)*((x[x>=t]).sum())**2

# sum_xy_t_1
theoretical['sum[t,1](x*y)'] = (
    a*theoretical['sum[t,1](x)']
    +b1*theoretical['sum[t,s](x**2)']
    +b2*theoretical['sum]s,1](x**2)']
    -s*(b2-b1)*theoretical['sum]s,1](x)']
)
empirical['sum[t,1](x*y)'] = (x[x>=t]*y[x>=t]).sum()

#sum_y_t_1
theoretical['sum[t,1](y)'] = (n-t*n+1)*a-(n*(1-s))* s*(b2-b1)+b1*theoretical['sum[t,s](x)']+b2*theoretical['sum]s,1](x)']
empirical['sum[t,1](y)'] = y[x>=t].sum()

#avg_y_t_1
theoretical['avg[t,1](y)'] = ((n-t*n+1)*a-(n*(1-s))* s*(b2-b1)+b1*theoretical['sum[t,s](x)']+b2*theoretical['sum]s,1](x)'])/(n-t*n+1)
empirical['avg[t,1](y)'] = y[x>=t].mean()

theoretical['beta[t,1]'] = (theoretical['sum[t,1](x*y)']-(n-t*n+1)*theoretical['avg[t,1](y)']*theoretical['avg[t,1](x)'])/theoretical['denominator(beta[t,1])']
empirical['beta[t,1]'] = np.polyfit(x[x>=t], y[x>=t], 1)[0]

for key in theoretical.keys():
    if abs(theoretical[key]-empirical[key]) > 1e-8:
        print(f'ERROR: {key}')
