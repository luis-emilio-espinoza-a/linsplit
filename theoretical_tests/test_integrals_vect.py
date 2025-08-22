import matplotlib.pyplot as plt
import numpy as np

M = np.array([[1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0],
[1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0],
[1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0],
[1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 0],
[1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]], dtype=np.float64)
C = np.array((81, 108, 54, 12, 1), dtype=np.float64).reshape(1,5)
exponents = np.arange(7, 12, dtype=np.float64).reshape(1,5)

def check_inputs(*args):
    # Revisa que los tipos de los inputs sean int, float o ndarray, que los arreglos sean columnas nx1
    # y que todos los array tengan la misma forma.
    ndim = []
    for value in args:
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(f'Argument is {type(value)}, expected to be {np.ndarray}, {int}, or {float}')
        if isinstance(value, np.ndarray):
            if len(value.shape) <= 1 or len(value.shape)>2:
                raise TypeError('Arrays expected to be (n,1) shape')
            if len(value.shape) == 2 and value.shape[1] != 1:
                raise TypeError('Arrays expected to be (n,1) shape')
            ndim.append(value.shape[0])
    if len(set(ndim))>1:
        raise TypeError('Arrays expected to be same shape')

def scalar_to_column(*args):
    colunms = [np.array(i).reshape(-1,1) if not isinstance(i, np.ndarray) else i for i in args]
    return colunms[0] if len(colunms) == 1 else colunms

def isndarray(*args):
    condition_array_list = [isinstance(i, np.ndarray) for i in args]
    return condition_array_list[0] if len(condition_array_list) == 1 else condition_array_list

def norm_shapes(*args):
    norm_vecs = []
    for value in args:
        if isinstance(value, np.ndarray):
            N = value.shape[0]
            break
    else:
        return args[0] if len(args) == 1 else args
    for value in args:
        if not isinstance(value, np.ndarray):
            norm_vecs.append(value*np.ones((N,1), dtype=np.float64))
        else:
            norm_vecs.append(value)
    return norm_vecs[0] if len(norm_vecs) == 1 else norm_vecs

def int_1_vect(k, c):
    # (A) integral (3*x+c)**4/(1-x)**12 de 0 a k.
    check_inputs(k, c)
    condition_array = isndarray(k, c)
    V_left = (((1-k)**(-exponents)-1)/ exponents) # size (N,5)
    c = scalar_to_column(c)
    A = np.stack([(3+c)**0, -(3+c)**1, (3+c)**2, -(3+c)**3, (3+c)**4], axis=1).squeeze(-1) # size (N,5)
    result = np.einsum('ij,kj,ij->i', A, C, V_left).reshape(-1, 1)
    return result if any(condition_array) else result[0][0]

def int_2_vect(k, c):
    # (B) integral (3*x+c)**4/x**12 de k a 1.
    check_inputs(k, c)
    condition_array = isndarray(k, c)
    V_right = (k**(-exponents)-1)/exponents
    c = scalar_to_column(c)
    A = np.stack([c**0, c**1, c**2, c**3, c**4], axis=1).squeeze(-1)
    result = np.einsum('ij,kj,ij->i', A, C, V_right).reshape(-1, 1)
    return result if any(condition_array) else result[0][0]

def int_3_vect(k, c1, c2):
# (C) integral (3*x+c1)**2*(3*x+c2)**2/(1-x)**12 de 0 a k.
    check_inputs(k, c1, c2)
    condition_array = isndarray(k, c1, c2)
    V_left = (((1-k)**(-exponents)-1)/ exponents) # size (N,5)
    if any(condition_array[1:]):
        c1, c2 = norm_shapes(c1, c2)
        constant = np.ones_like(c1, dtype=np.float64)
    else:
        c1, c2, constant = scalar_to_column(c1, c2, 1)
    A = np.stack([constant, -(c1+c2)/2-3, 3*(c1+c2)+(c1**2+c2**2)/6+2*c1*c2/3+9,
    -3*(c1**2+c2**2)/2- c1*c2*(c1+c2)/2-6*c1*c2-27*(c1+c2)/2-27,
    c1**2*c2**2+9*(c1**2+c2**2)+6*c1*c2*(c1 + c2)+36*c1*c2+54*(c1+c2)+81], axis=1).squeeze(-1)
    result = np.einsum('ij,kj,ij->i', A, C, V_left).reshape(-1, 1)
    return result if any(condition_array) else result[0][0]

def int_4_vect(k, c1, c2):
# (D) integral (3*x+c1)**2*(3*x+c2)**2/x**12 de k a 1.
    check_inputs(k, c1, c2)
    condition_array = isndarray(k, c1, c2)
    V_right = (k**(-exponents)-1)/exponents
    if any(condition_array[1:]):
        c1, c2 = norm_shapes(c1, c2)
        constant = np.ones_like(c1, dtype=np.float64)
    else:
        c1, c2, constant = scalar_to_column(c1, c2, 1)
    A = np.stack([constant, (c1+c2)/2, (c1**2+4*c1*c2+c2**2)/6, c1*c2*(c1+c2)/2, c1**2*c2**2], axis=1).squeeze(-1)
    result = np.einsum('ij,kj,ij->i', A, C, V_right).reshape(-1, 1)
    return result if any(condition_array) else result[0][0]

def int_5_vect(k, l, c1, c2):
    # (E) integral (3*x+c1)**2*(3*x+c2)**2/(x**6*(1-x)**6) de k a l
    check_inputs(k, l, c1, c2)
    condition_array = isndarray(k, l, c1, c2)
    if any(condition_array[:2]):
        k, l = norm_shapes(k, l)
        N = k.shape[0]
    else:
        N = 1
    a = (1-k)/k
    b = (1-l)/l
    m = np.arange(11, dtype=np.float64).reshape(-1, 1) 
    mask = (m != 5)
    V_midd = np.zeros((N, 11), dtype=np.float64)
    m_f1 = m[mask]
    V_midd[:, mask.ravel()] = (1/(m_f1-5))*(a**(m_f1-5)-b**(m_f1 - 5))
    V_midd[:,[5]] = np.log(a) - np.log(b)
    if any(condition_array[2:]):
        c1, c2 = norm_shapes(c1, c2)
        constant = np.ones_like(c1, dtype=np.float64)
    else:
        c1, c2, constant = scalar_to_column(c1, c2, 1)
    A = np.stack([constant, (c1+c2)/2, (c1**2+4*c1*c2+c2**2)/6, c1*c2*(c1+c2)/2, c1**2*c2**2], axis=1).squeeze(-1)
    result = np.einsum('ij,ij->i', C*A, V_midd@M.T).reshape(-1,1)
    return result if any(condition_array) else result[0][0]

def mse_d2(T, s, b1, b2):
    n = len(T)

    mask = (s<=T).ravel()
    MSE_D2 = np.zeros((n,1))
    b1_t = np.zeros((n,1))
    b2_t = np.zeros((n,1))

    b1_t[mask] = (b1*s**2*(3*T[mask]-2*s)+b2*(T[mask]-s)**2*(T[mask]+2*s))/T[mask]**3
    b1_t[~mask] = b1
    b2_t[~mask] = (b1*(s-T[~mask])**2*(3-2*s-T[~mask])+b2*(1-s)**2*(1+2*s-3*T[~mask]))/(1-T[~mask])**3
    b2_t[mask] = b2
    fdm = (b2-b1)**2*(1-s)**4
    fdp = (b2-b1)**2*s**4
    fdl = (b2_t-b1_t)**2*(1-T)**4
    fdr = (b2_t-b1_t)**2*T**4

    MSE_D2 = np.zeros((n,1))

    int_1a = int_1_vect(T[~mask], -2*T[~mask]-1)
    int_1b = int_1_vect(T[~mask], -2*s-1)
    int_1c = int_1_vect(s, -2*s-1)
    int_2a = int_2_vect(T[~mask], -2*T[~mask])
    int_2b = int_2_vect(s, -2*T[~mask])
    int_2c = int_2_vect(s, -2*s)
    int_3a = int_3_vect(T[~mask], -2*s-1, -2*T[~mask]-1)
    int_4a = int_4_vect(s, -2*T[~mask], -2*s)
    int_5a = int_5_vect(T[~mask], s, -2*T[~mask], -2*s-1)
    a = fdl[~mask]**2*int_1a-2*fdl[~mask]*fdm*int_3a+fdm**2*int_1b
    b = fdr[~mask]**2*(int_2a-int_2b)-2*fdr[~mask]*fdm*int_5a+fdm**2*(int_1c-int_1b)
    c = fdr[~mask]**2*int_2b-2*fdr[~mask]*fdp*int_4a+fdp**2*int_2c
    MSE_D2[~mask] = 1/T[~mask]*a+1/(1-T[~mask])*(b+c)

    int_1a = int_1_vect(s, -2*T[mask]-1)
    int_1b = int_1_vect(s, -2*s-1)
    int_1c = int_1_vect(T[mask], -2*T[mask]-1)
    int_2a = int_2_vect(s, -2*s)
    int_2b = int_2_vect(T[mask], -2*s)
    int_2c = int_2_vect(T[mask], -2*T[mask])
    int_3a = int_3_vect(s, -2*s-1, -2*T[mask]-1)
    int_4a = int_4_vect(T[mask], -2*T[mask], -2*s)
    int_5a = int_5_vect(s, T[mask], -2*T[mask]-1, -2*s)
    a = fdl[mask]**2*int_1a-2*fdl[mask]*fdm*int_3a+fdm**2*int_1b
    b = fdl[mask]**2*(int_1c-int_1a)-2*fdl[mask]*fdp*int_5a+fdp**2*(int_2a-int_2b)
    c = fdr[mask]**2*int_2c-2*fdr[mask]*fdp*int_4a+fdp**2*int_2b
    MSE_D2[mask] = 1/T[mask]*(a+b)+1/(1-T[mask])*c

    return MSE_D2

def mse_d2_est(X, Y):

    beta_left = placket_vect(X, Y)
    beta_right = placket_vect(X[::-1], Y[::-1])
    beta_right = beta_right[::-1]
    D2 = (beta_right-beta_left)**2

    X_fin = X[k:-k].copy()
    N_fin = len(X_fin)
    D2_fin = D2[k:-k].copy()
    beta_left_fin = beta_left[k:-k].copy()
    beta_right_fin = beta_right[k:-k].copy()
    MSE_D2_est = np.zeros((N_fin,1))

    D2_sum = np.cumsum(D2_fin**2, axis=0)
    X2_mX12_sum = np.cumsum(X_fin**2/(1-X_fin)**12, axis=0)
    X3_mX12_sum = np.cumsum(X_fin**3/(1-X_fin)**12, axis=0)
    X4_mX12_sum = np.cumsum(X_fin**4/(1-X_fin)**12, axis=0)
    X_mX12_sum = np.cumsum(X_fin/(1-X_fin)**12, axis=0)
    one_mX12_sum = np.cumsum(1/(1-X_fin)**12, axis=0)
    D_mX6_sum = np.cumsum(D2_fin/(1-X_fin)**6, axis=0)
    DX_mX6_sum = np.cumsum(D2_fin*X_fin/(1-X_fin)**6, axis=0)
    DX2_mX6_sum = np.cumsum(D2_fin*X_fin**2/(1-X_fin)**6, axis=0)
    D2_sum_inv = np.cumsum((D2_fin**2)[::-1], axis=0)[::-1]
    D_X4_sum_inv = np.cumsum((D2_fin/X_fin**4)[::-1], axis=0)[::-1]
    D_X5_sum_inv = np.cumsum((D2_fin/X_fin**5)[::-1], axis=0)[::-1]
    D_X6_sum_inv = np.cumsum((D2_fin/X_fin**6)[::-1], axis=0)[::-1]
    one_X8_sum_inv = np.cumsum((1/X_fin**8)[::-1], axis=0)[::-1]
    one_X9_sum_inv = np.cumsum((1/X_fin**9)[::-1], axis=0)[::-1]
    one_X10_sum_inv = np.cumsum((1/X_fin**10)[::-1], axis=0)[::-1]
    one_X11_sum_inv = np.cumsum((1/X_fin**11)[::-1], axis=0)[::-1]
    one_X12_sum_inv = np.cumsum((1/X_fin**12)[::-1], axis=0)[::-1]

    a_left = 3*(beta_left_fin-beta_right_fin)*(1-X_fin)**2
    b_left = (beta_left_fin-beta_right_fin)*(2*X_fin**3-3*X_fin**2+1)
    a_right = 3*X_fin**2*(beta_right_fin-beta_left_fin)
    b_right = 2*X_fin**3*(beta_right_fin-beta_left_fin)

    sum_left = (
        D2_sum
        +a_left**4*X4_mX12_sum
        -4*a_left**3*b_left*X3_mX12_sum
        +6*a_left**2*b_left**2*X2_mX12_sum
        -4*a_left*b_left**3*X_mX12_sum
        +b_left**4*one_mX12_sum
        -2*a_left**2*DX2_mX6_sum
        +4*a_left*b_left*DX_mX6_sum
        -2*b_left**2*D_mX6_sum
    )

    sum_right=(
        D2_sum_inv
        -2*a_right**2*D_X4_sum_inv
        +4*a_right*b_right*D_X5_sum_inv
        -2*b_right**2*D_X6_sum_inv
        +a_right**4*one_X8_sum_inv
        -4*a_right**3*b_right*one_X9_sum_inv
        +6*a_right**2*b_right**2*one_X10_sum_inv
        -4*a_right*b_right**3*one_X11_sum_inv
        +b_right**4*one_X12_sum_inv
    )

    DEN = np.linspace(1,N_fin-1,N_fin-1).reshape(-1,1)
    MSE_D2_est = sum_left[:-1]/DEN+sum_right[1:]/DEN[::-1]

    return MSE_D2_est

def mse_d2_est(X, Y, k):

    beta_left = placket_vect(X, Y)
    beta_right = placket_vect(X[::-1], Y[::-1])
    beta_right = beta_right[::-1]
    D2 = (beta_right-beta_left)**2

    X_fin = X[k:-k].copy()
    N_fin = len(X_fin)
    D2_fin = D2[k:-k].copy()
    beta_left_fin = beta_left[k:-k].copy()
    beta_right_fin = beta_right[k:-k].copy()
    MSE_D2_est = np.zeros((N_fin,1))

    D2_sum = np.cumsum(D2_fin**2, axis=0)
    X2_mX12_sum = np.cumsum(X_fin**2/(1-X_fin)**12, axis=0)
    X3_mX12_sum = np.cumsum(X_fin**3/(1-X_fin)**12, axis=0)
    X4_mX12_sum = np.cumsum(X_fin**4/(1-X_fin)**12, axis=0)
    X_mX12_sum = np.cumsum(X_fin/(1-X_fin)**12, axis=0)
    one_mX12_sum = np.cumsum(1/(1-X_fin)**12, axis=0)
    D_mX6_sum = np.cumsum(D2_fin/(1-X_fin)**6, axis=0)
    DX_mX6_sum = np.cumsum(D2_fin*X_fin/(1-X_fin)**6, axis=0)
    DX2_mX6_sum = np.cumsum(D2_fin*X_fin**2/(1-X_fin)**6, axis=0)
    D2_sum_inv = np.cumsum((D2_fin**2)[::-1], axis=0)[::-1]
    D_X4_sum_inv = np.cumsum((D2_fin/X_fin**4)[::-1], axis=0)[::-1]
    D_X5_sum_inv = np.cumsum((D2_fin/X_fin**5)[::-1], axis=0)[::-1]
    D_X6_sum_inv = np.cumsum((D2_fin/X_fin**6)[::-1], axis=0)[::-1]
    one_X8_sum_inv = np.cumsum((1/X_fin**8)[::-1], axis=0)[::-1]
    one_X9_sum_inv = np.cumsum((1/X_fin**9)[::-1], axis=0)[::-1]
    one_X10_sum_inv = np.cumsum((1/X_fin**10)[::-1], axis=0)[::-1]
    one_X11_sum_inv = np.cumsum((1/X_fin**11)[::-1], axis=0)[::-1]
    one_X12_sum_inv = np.cumsum((1/X_fin**12)[::-1], axis=0)[::-1]

    a_left = 3*(beta_left_fin-beta_right_fin)*(1-X_fin)**2
    b_left = (beta_left_fin-beta_right_fin)*(2*X_fin**3-3*X_fin**2+1)
    a_right = 3*X_fin**2*(beta_right_fin-beta_left_fin)
    b_right = 2*X_fin**3*(beta_right_fin-beta_left_fin)

    sum_left = (
        D2_sum
        +a_left**4*X4_mX12_sum
        -4*a_left**3*b_left*X3_mX12_sum
        +6*a_left**2*b_left**2*X2_mX12_sum
        -4*a_left*b_left**3*X_mX12_sum
        +b_left**4*one_mX12_sum
        -2*a_left**2*DX2_mX6_sum
        +4*a_left*b_left*DX_mX6_sum
        -2*b_left**2*D_mX6_sum
    )

    sum_right=(
        D2_sum_inv
        -2*a_right**2*D_X4_sum_inv
        +4*a_right*b_right*D_X5_sum_inv
        -2*b_right**2*D_X6_sum_inv
        +a_right**4*one_X8_sum_inv
        -4*a_right**3*b_right*one_X9_sum_inv
        +6*a_right**2*b_right**2*one_X10_sum_inv
        -4*a_right*b_right**3*one_X11_sum_inv
        +b_right**4*one_X12_sum_inv
    )

    DEN = np.linspace(1,N_fin-1,N_fin-1).reshape(-1,1)
    MSE_D2_est = sum_left[:-1]/DEN+sum_right[1:]/DEN[::-1]

    return MSE_D2_est, beta_left_fin, beta_right_fin

n = 1000
s = 0.7
b1 = 1
b2 = -1
T = np.linspace(1/n, 1-1/n, n-1, dtype=np.float64).reshape(-1,1)

MSE_D2 = mse_d2(T, s, b1, b2)

plt.plot(T, MSE_D2)
plt.savefig('MSE_t.png')
plt.clf()
