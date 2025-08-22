from sympy import symbols, simplify, expand, collect, limit, factor

theoretical ={}

# CASE S<T

n, s, t, a, b1, b2 = symbols('n, s, t, a, b1, b2')

theoretical['lenght(]0, s[)'] = n*s-1
theoretical['lenght(]0, t])'] = n*t
theoretical['lenght([s, t])'] = t*n-s*n+1
theoretical['sum[s,t](x)'] = (t+s)*(1+n*(t-s))/2
theoretical['sum[s,t](x**2)'] = (t*(t*n+1)*(2*t*n+1)-s*(s*n-1)*(2*s*n-1))/(6*n)
theoretical['sum]0,t](x)'] = (t*n+1)*t/2
theoretical['sum]0,t](x**2)'] = (t*(t*n+1)*(2*t*n+1))/(6*n)
theoretical['sum]0,s[(x)'] = s*(s*n-1)/2
theoretical['sum]0,s[(x**2)'] = (s*(s*n-1)*(2*s*n-1))/(6*n)
theoretical['avg]0,t](x)'] = (t+1/n)/2
theoretical['denominator(beta]0,t])'] = t*((t*n)**2-1)/12/n
theoretical['sum]0,t](x*y)'] = (
    a*theoretical['sum]0,t](x)']+
    b1*theoretical['sum]0,s[(x**2)']+
    b2*theoretical['sum[s,t](x**2)']-
    s*(b2-b1)*theoretical['sum[s,t](x)']
)
theoretical['sum]0,t](y)'] = t*n*a-(t*n-s*n+1)*s*(b2-b1)+b1*theoretical['sum]0,s[(x)']+b2*theoretical['sum[s,t](x)']
theoretical['avg]0,t](y)'] = theoretical['sum]0,t](y)']/(n*t)
theoretical['beta]0,t]'] = (theoretical['sum]0,t](x*y)']-t*n*theoretical['avg]0,t](y)']*theoretical['avg]0,t](x)'])/theoretical['denominator(beta]0,t])']

numerator = theoretical['sum]0,t](x*y)']-t*n*theoretical['avg]0,t](y)']*theoretical['avg]0,t](x)']
numerator_expand = expand(numerator)
numerator_collect = collect(numerator_expand, n, evaluate=False)

numerator_n = simplify(numerator_collect.get(n**1, 0))*n
numerator_c = simplify(numerator_collect.get(n**0, 0))
numerator_dn = simplify(numerator_collect.get(n**-1, 0))/n

par1 = numerator_n/theoretical['denominator(beta]0,t])']
par2 = numerator_c/theoretical['denominator(beta]0,t])']
par3 = numerator_dn/theoretical['denominator(beta]0,t])']

limit1 = simplify(limit(par1, n, float('inf')))
limit2 = simplify(limit(par2, n, float('inf')))
limit3 = simplify(limit(par3, n, float('inf')))

exp1 = limit1+limit2+limit3
denominator = t**3
exp2 = collect(exp1*denominator, [b1, b2])
factor_b1 = factor(collect(exp2, b1, evaluate=False)[b1])
factor_b2 = factor(collect(exp2, b2, evaluate=False)[b2])
beta_left = (b1*factor_b1+b2*factor_b2)/denominator

# CASE S>T

theoretical = {}

theoretical['sum[t,s](x)'] = (s+t)*(1+n*(s-t))/2
theoretical['sum[t,s](x**2)'] = (s*(s*n+1)*(2*s*n+1)-t*(t*n-1)*(2*t*n-1))/(6*n)
theoretical['sum[t,1](x)'] = (1+t)*(1+n-t*n)/2
theoretical['sum[t,1](x**2)'] = ((n+1)*(2*n+1)-t*(t*n-1)*(2*t*n-1))/(6*n)
theoretical['sum]s,1](x)'] = (n+s*n+1)*(1-s)/2
theoretical['sum]s,1](x**2)'] = ((n+1)*(2*n+1)-s*(s*n+1)*(2*s*n+1))/(6*n)
theoretical['avg[t,1](x)'] = (1+t)/2
theoretical['denominator(beta[t,1])'] = (1-t)*(n*t-n-1)*(n*t-n-2)/(12*n)
theoretical['sum[t,1](x*y)'] = (
    a*theoretical['sum[t,1](x)']
    +b1*theoretical['sum[t,s](x**2)']
    +b2*theoretical['sum]s,1](x**2)']
    -s*(b2-b1)*theoretical['sum]s,1](x)']
)
theoretical['sum[t,1](y)'] = (n-t*n+1)*a-(n*(1-s))* s*(b2-b1)+b1*theoretical['sum[t,s](x)']+b2*theoretical['sum]s,1](x)']
theoretical['avg[t,1](y)'] = ((n-t*n+1)*a-(n*(1-s))* s*(b2-b1)+b1*theoretical['sum[t,s](x)']+b2*theoretical['sum]s,1](x)'])/(n-t*n+1)
theoretical['beta[t,1]'] = (theoretical['sum[t,1](x*y)']-(n-t*n+1)*theoretical['avg[t,1](y)']*theoretical['avg[t,1](x)'])/theoretical['denominator(beta[t,1])']

numerator = theoretical['sum[t,1](x*y)']-(n-t*n+1)*theoretical['avg[t,1](y)']*theoretical['avg[t,1](x)']
numerator_expand = expand(numerator)
numerator_collect = collect(numerator_expand, n, evaluate=False)

numerator_n = simplify(numerator_collect.get(n**1, 0))*n
numerator_c = simplify(numerator_collect.get(n**0, 0))
numerator_dn = simplify(numerator_collect.get(n**-1, 0))/n

par1 = numerator_n/theoretical['denominator(beta[t,1])']
par2 = numerator_c/theoretical['denominator(beta[t,1])']
par3 = numerator_dn/theoretical['denominator(beta[t,1])']

limit1 = simplify(limit(par1, n, float('inf')))
limit2 = simplify(limit(par2, n, float('inf')))
limit3 = simplify(limit(par3, n, float('inf')))

exp1 = limit1+limit2+limit3
denominator = (t-1)**3
exp2 = collect(exp1*denominator, [b1, b2])
factor_b1 = factor(collect(exp2, b1, evaluate=False)[b1])
factor_b2 = factor(collect(exp2, b2, evaluate=False)[b2])
beta_right = (b1*factor_b1+b2*factor_b2)/denominator

print('beta_left:', beta_left, '\n', 'beta_right:', beta_right)