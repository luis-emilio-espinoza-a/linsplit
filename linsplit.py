import numpy as np

def beta_vect(X, Y, gap_left = 2, gap_right = 0):
    """
    Rolling slope (beta) for every point in X vs Y.

    gap_left:  skip this many points from the start
    gap_right: skip this many points from the end

    Returns:
        numpy array: same shape as X, 0 where skipped, slope elsewhere
    """
    X_len = len(X)
    start = 0 if gap_left <= 0 else gap_left
    end = X_len if gap_right <= 0 else -gap_right
    Sx = np.cumsum(X, axis=0)[start:end]
    Sy = np.cumsum(Y, axis=0)[start:end]
    Sx2 = np.cumsum(np.square(X), axis=0)[start:end]
    Sy2 = np.cumsum(np.square(Y), axis=0)[start:end]
    Sxy = np.cumsum(np.multiply(X,Y), axis=0)[start:end]
    N = np.linspace(1, X_len, X_len).reshape(-1,1)[start:end]
    Mx = Sx/N
    My = Sy/N
    _beta = (Sxy-N*Mx*My)/(Sx2-N*Mx**2)
    beta = np.zeros_like(X)
    beta[start:end] = _beta
    return beta

def r2_no_intercept(X, Y, gap_left = 1, gap_right = 0, cutgap=True):
    """
    Rolling R² (no intercept) for every point in X vs Y.

    gap_left:  skip this many points from the start
    gap_right: skip this many points from the end
    cutgap:    if False, zeros are padded instead of slicing

    Returns:
        numpy array: same shape as X, R² values (or 0 where skipped)
    """
    X_len = len(X)
    start = 0 if gap_left <= 0 else gap_left
    end = X_len if gap_right <= 0 else -gap_right
    Sx = np.cumsum(X, axis=0)[start:end]
    Sy = np.cumsum(Y, axis=0)[start:end]
    Sxy = np.cumsum(X * Y, axis=0)[start:end]
    Sx2 = np.cumsum(np.square(X), axis=0)[start:end]
    Sy2 = np.cumsum(np.square(Y), axis=0)[start:end]
    N = np.arange(1, len(X) + 1).reshape(-1, 1)[start:end]
    #_beta = Sxy/Sx2
    SSE = Sy2-Sxy**2/Sx2
    SST_centered = Sy2-(Sy**2)/N
    _r2 = 1-SSE/SST_centered
    if cutgap:
        #beta, r2 = _beta, _r2
        r2 = _r2
    else:
        # beta, r2 = np.zeros_like(X), np.zeros_like(X)
        # beta[start:end], r2[start:end] = _beta, _r2
        r2 = np.zeros_like(X)
        r2[start:end] = _r2
    # return beta, r2
    return r2

def r2_no_intercept_vect(X, Y):
    """
    Vectorized slope for SLR without intercept, one sample per row.

    X, Y: 2-D arrays, shape (samples, features)

    Returns:
        numpy array (samples,1): slope for each row
    """
    Sy = Y.sum(axis=1)
    Sxy = (X*Y).sum(axis=1)
    Sx2 = (X**2).sum(axis=1)
    Sy2 = (Y**2).sum(axis=1)
    Nc, N = X.shape
    beta = Sxy/Sx2
    return beta.reshape(Nc,1)

def calc_perc_neighbor(metric, wlen_neighbor):
    """
    % of neighbors within wlen_neighbor on each side that are larger than current value.

    metric: 1-D array
    wlen_neighbor: half-window size (int)

    Returns:
        1-D array: neighbor percentage for every point
    """

    metric = metric.ravel()
    metric_R = np.concat([metric, np.zeros(wlen_neighbor)])
    # neighbor in right
    R = np.ascontiguousarray(metric_R[1:])
    R = np.lib.stride_tricks.sliding_window_view(R, wlen_neighbor).copy()
    metric_L = np.concat([np.zeros(wlen_neighbor), metric])
    # neighbor in left
    L = np.ascontiguousarray(metric_L[:-1])
    L = np.lib.stride_tricks.sliding_window_view(L, wlen_neighbor).copy()
    # neighbor percentage greater than the value in the metric. 
    perc_neighbor_vect = (
        (metric.reshape(-1,1)<R).sum(axis=1).reshape(-1,1)+
        (metric.reshape(-1,1)<L).sum(axis=1).reshape(-1,1)
        )/(2*wlen_neighbor)
    return perc_neighbor_vect

def first_subset_condition(vect_bool):
    """
    First consecutive True block in a boolean array.

    vect_bool: 1-D bool array

    Returns:
        tuple (start, end) or -1 if no True found
    """
    idx1 = int(np.argmax(vect_bool))
    if not vect_bool[idx1]:
        return -1
    vect_bool_inv = np.logical_not(vect_bool[idx1:])
    if not np.any(vect_bool_inv):
        return idx1, len(vect_bool)-1
    idx2_tmp = int(np.argmax(vect_bool_inv))
    idx2 = idx2_tmp - 1 if (vect_bool_inv[idx2_tmp] and idx2_tmp > 0) else 0
    return idx1, idx1 +idx2

def get_first_signature_pos(X, Y, wlen_signature, wlen_neighbor, tol_neighbor, tol_signature):
    """
    Finds the first “signature” (possible change-point) in the signal.

    X, Y: full data vectors
    wlen_signature: half-window to build the signature test
    wlen_neighbor:  half-window for local-min refinement
    tol_neighbor:   min % of smaller neighbors to keep candidate
    tol_signature:  tolerance for residual band test

    Returns:
        (s_candidate, end_of_signature): indices or (-1,-1) if nothing found
    """
    X_len = X.shape[0]
    if X_len <= 2*wlen_signature:
        # No enough data to sliding_window_view
        return -1, None

    beta = beta_vect(X, Y)

    # The Hermite Signature function
    signature = lambda S, T: 2*S**3/T**3-3*S**2/T**2+1

    # Cálculo matrices L (left beta(t): x<=s), R (right beta(t) x>s)
    # S (x=s), XR (right X, x>s), B1 (beta_1, x=s).
    L = beta[:X_len-wlen_signature].ravel()
    L = np.ascontiguousarray(L) # order='C' muy verboso.
    L = np.lib.stride_tricks.sliding_window_view(L, wlen_signature).copy()
    R = beta[wlen_signature:].ravel()
    R = np.ascontiguousarray(R)
    R = np.lib.stride_tricks.sliding_window_view(R, wlen_signature).copy()
    XR = X[wlen_signature:].ravel()
    XR = np.ascontiguousarray(XR)
    XR = np.lib.stride_tricks.sliding_window_view(XR, wlen_signature).copy()
    S = X[wlen_signature-1:-wlen_signature].copy()
    B1 = beta[wlen_signature-1:-wlen_signature].copy()

    L_mean = np.mean(L, axis=1).reshape(X_len-2*wlen_signature+1,1) # Cambiar nombre

    # [t]heoretical vs [o]bserved signature with r2-score
    # Ojo, acá se puede acotar la función r2_no_intercept_vect sólo a betas 
    t_signature = signature(S, XR)
    o_signature = R-B1
    beta_sig = r2_no_intercept_vect(t_signature, o_signature) 

    # Mean squared error
    residual_left = L-L_mean
    residual_right = o_signature-t_signature*beta_sig
    mse1 = (residual_left**2).sum(axis=1)
    mse2 = (residual_right**2).sum(axis=1)
    mse = (mse1+mse2)/(2*wlen_signature-2)

    # Find min value in mse
    perc_neighbor = calc_perc_neighbor(mse, wlen_neighbor)
    if perc_neighbor.max() < tol_neighbor: # Can't find anything
        return -1, -1
    idx_ini, idx_end = first_subset_condition(perc_neighbor>=tol_neighbor)
    if idx_end-idx_ini+1 < wlen_neighbor:
        idx = np.argmin(mse[idx_ini:idx_ini+wlen_neighbor])+idx_ini
    else:
        idx = np.argmin(mse[idx_ini:idx_end+1])+idx_ini

    # Reconstruye t_signature y o_signature para todos los datos a la derecha restantes.
    o_signature_new = beta[idx+wlen_signature:]-B1[idx] # R[idx].reshape(-1,1)==beta[idx+wlen_signature:][:30]
    S_new = S[idx][0] # S[idx][0]==X[idx+wlen_signature-1][0]
    XR_new = X[idx+wlen_signature:] # XR[idx].reshape(-1,1)==X[idx+wlen_signature:][:30]
    t_signature_new = signature(S_new, XR_new)
    beta_sig_new = beta_sig[idx][0]

    # Residual simple test
    # Residues observed in new [:wlen_signature] data
    e_p = o_signature_new[:wlen_signature]-t_signature_new[:wlen_signature]*beta_sig_new
    # Residues observed in [wlen_signature+1:] data
    e_est = o_signature_new-t_signature_new*beta_sig_new
    # Residual bandwith
    ep_min, ep_max = e_p.min(), e_p.max()
    e_inf = (ep_max+ep_min)/2-(1+tol_signature)*(ep_max-ep_min)/2
    e_sup = (ep_max+ep_min)/2+(1+tol_signature)*(ep_max-ep_min)/2
    # get first out of the rank. At wlen_signature data pass the simple test.
    idx_new = np.argmax(np.logical_or(e_est<e_inf, e_est>e_sup))
    # if idx_new 
    # if idx_new>idx+wlen_signature-1:
    #     idx_new = idx+wlen_signature-1
    # Si gap>wlen_signature, podría darse el caso borde de que el changepoint esté en el límite.
    s_candidate = idx+wlen_signature-1
    end_of_signature = s_candidate+idx_new

    return int(s_candidate), int(end_of_signature)

def r2_mses(MSE_D2_est, MSE_D2):
    SS_res = np.sum(np.square(MSE_D2_est-MSE_D2))
    SS_tot = np.sum(np.square(MSE_D2_est))
    return 1-(SS_res/SS_tot)

def norm_ppf(p):
    "Normal inverse CDF using Acklam's algorithm."
    a_1 = -3.969683028665376e+01
    a_2 =  2.209460984245205e+02
    a_3 = -2.759285104469687e+02
    a_4 =  1.383577518672690e+02
    a_5 = -3.066479806614716e+01
    a_6 =  2.506628277459239e+00
    b_1 = -5.447609879822406e+01
    b_2 =  1.615858368580409e+02
    b_3 = -1.556989798598866e+02
    b_4 =  6.680131188771972e+01
    b_5 = -1.328068155288572e+01
    c_1 = -7.784894002430293e-03
    c_2 = -3.223964580411365e-01
    c_3 = -2.400758277161838e+00
    c_4 = -2.549732539343734e+00
    c_5 =  4.374664141464968e+00
    c_6 =  2.938163982698783e+00
    d_1 =  7.784695709041462e-03
    d_2 =  3.224671290700398e-01
    d_3 =  2.445134137142996e+00
    d_4 =  3.754408661907416e+00
    p_low  = 0.02425
    p_high = 1 - p_low
    if 0 < p < p_low:
        q = np.sqrt(-2*np.log(p))
        x = (((((c_1*q+c_2)*q+c_3)*q+c_4)*q+c_5)*q+c_6) / ((((d_1*q+d_2)*q+d_3)*q+d_4)*q+1)
    if p_low <= p <= p_high:
        q = p - 0.5
        r = q*q
        x = (((((a_1*r+a_2)*r+a_3)*r+a_4)*r+a_5)*r+a_6)*q / (((((b_1*r+b_2)*r+b_3)*r+b_4)*r+b_5)*r+1)
    if p_high < p < 1:
        q = np.sqrt(-2*np.log(1-p))
        x = -(((((c_1*q+c_2)*q+c_3)*q+c_4)*q+c_5)*q+c_6) / ((((d_1*q+d_2)*q+d_3)*q+d_4)*q+1)
    return x

def t_ppf(p, v):
    "t-student inverse CDF using Cornish Fisher expansion."
    z = norm_ppf(p)
    t = z
    t +=  (z**3+z)/(4*v)
    t +=  (5*z**5+16*z**3+3*z)/(96*v**2)
    t +=  (3*z**7+19*z**5+17*z**3-15*z)/(384*v**3)
    t +=  (79*z**9+776*z**7+1482*z**5-1920*z**3-945*z)/(92160*v**4)
    t +=  (9*z**11+113*z**9+310*z**7-594*z**5-255*z**3+5985*z)/(122880*v**5)
    t += (1065*z**13+15448*z**11+48821*z**9-82440*z**7+616707*z**5+6667920*z**3+2463615*z)/(185794560*v**6)
    t += (339*z**15+6891*z**13+41107*z**11+113891*z**9+1086849*z**7+5639193*z**5-18226215*z**3-111486375*z)/(743178240*v**7)
    t += (9159*z**17+296624*z**15+3393364*z**13+16657824*z**11+27817290*z**9-591760080*z**7-9178970220*z**5-42618441600*z**3-14223634425*z)/(356725555200*v**8)
    return t

def dtv_ppf(p, v):
    "t-student inverse CDF derivate using Cornish Fisher expansion."
    z = norm_ppf(p)
    dt_dv = -(z**3 + z)/(4*v**2)
    dt_dv += -(5*z**5 + 16*z**3 + 3*z)/(48*v**3)
    dt_dv += -(3*z**7 + 19*z**5 + 17*z**3 - 15*z)/(128*v**4)
    dt_dv += -(79*z**9 + 776*z**7 + 1482*z**5 - 1920*z**3 - 945*z)/(23040*v**5)
    dt_dv += -(9*z**11 + 113*z**9 + 310*z**7 - 594*z**5 - 255*z**3 + 5985*z)/(24576*v**6)
    dt_dv += -(1065*z**13 + 15448*z**11 + 48821*z**9 - 82440*z**7 + 616707*z**5 + 6667920*z**3 + 2463615*z)/(30965760*v**7)
    dt_dv += -(339*z**15 + 6891*z**13 + 41107*z**11 + 113891*z**9 + 1086849*z**7 + 5639193*z**5 - 18226215*z**3 - 111486375*z)/(92897280*v**8)
    dt_dv += -(9159*z**17 + 296624*z**15 + 3393364*z**13 + 16657824*z**11 + 27817290*z**9 - 591760080*z**7 - 9178970220*z**5 - 42618441600*z**3 - 14223634425*z)/(44690694400*v**9)
    return dt_dv

def d_IC_len_k_empirical(X, Y, alpha = 0.05):
    """
    Empirical derivatives of CI width and MSE for k ≥ 5.

    X, Y: data vectors
    alpha: significance level

    Returns:
        (d_W_dk, d_MSE_k_dk, MSE_k, Beta_k, K): arrays starting at k=5
    """
    # Devuelve sólo data para k>=5
    n_gap = 2
    X_len = len(X)
    X_stand = (X-X.mean())/X.std()
    Sx_k = np.cumsum(X_stand, axis=0)[n_gap:]
    Sy_k = np.cumsum(Y, axis=0)[n_gap:]
    Sx2_k = np.cumsum(np.square(X_stand), axis=0)[n_gap:]
    Sy2_k = np.cumsum(np.square(Y), axis=0)[n_gap:]
    Sxy_k = np.cumsum(np.multiply(X_stand,Y), axis=0)[n_gap:]
    K = np.linspace(1, X_len, X_len, dtype=np.float64).reshape(-1,1)[n_gap:]
    Mx_k = Sx_k/K
    My_k = Sy_k/K
    Sxx_k = (Sx2_k-K*Mx_k**2)
    Beta_k = (Sxy_k-K*Mx_k*My_k)/Sxx_k
    Alpha_k = My_k-Beta_k*Mx_k
    MSE_k = (Sy2_k-Beta_k*Sxy_k-Alpha_k*Sy_k)/(K-2)
    MSE_k_prev = np.roll(MSE_k, shift=1, axis=0)
    MSE_k_prev[0] = np.nan
    Xm2_k = np.square(X_stand[n_gap:]-Mx_k)
    h_kk = 1/K+Xm2_k/Sxx_k
    d_MSE_k_dk = (MSE_k_prev-MSE_k)/K+2*((K-1)*MSE_k_prev-K*MSE_k)/(K-1)*h_kk
    d_Sxx_k_dk = K/(K-1)*Xm2_k
    d_t_ppf_dk = dtv_ppf(1-alpha/2, K-2)
    d_W_dk = (
        2*d_t_ppf_dk*np.sqrt(MSE_k/Sxx_k)
        +t_ppf(1-alpha/2, K-2)/np.sqrt(MSE_k*Sxx_k)*(d_MSE_k_dk-MSE_k/Sxx_k*d_Sxx_k_dk)
    )
    return d_W_dk[2:], d_MSE_k_dk[2:], MSE_k[2:], Beta_k[2:], K[2:]

def get_gap(X, Y, epsilon):
    d_W_dk, _, _, _, _ = d_IC_len_k_empirical(X, Y, alpha=0.05)
    idx = np.where(d_W_dk >= -epsilon)[0][0]
    gap = int(idx+5) # d_W_dk va de obs 1 a 
    return gap

# Definition constant of integrals (MSE_t).
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
    # Theoretical MSE_t
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

def mse_d2_est(X, Y, k):
    # Empitical MSE(x_i)
    beta_left = beta_vect(X, Y, gap_left = 2, gap_right = 2)
    beta_right = beta_vect(X[::-1], Y[::-1], gap_left = 2, gap_right = 2)
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

def piecewise_regression_sim(t, rates, changepoints, m, sigma_error):
    # Simula una recta con changepoints.
    if np.any(np.logical_or(changepoints <= t.min(), changepoints >= t.max())):
        raise Exception('Changepoints fuera de rango')
    # Cálculos
    t_nr = len(t)
    changepoints_nr = len(changepoints)
    # Cálculo vector delta y escalar k.
    delta = np.array([rates[i+1] - rates[i] for i in range(changepoints_nr)]).reshape(-1, 1)
    k = rates[0] # tasa de crecimiento al origen
    # Cálculo matriz A
    A = np.zeros((t_nr, changepoints_nr))
    for i, t_value in enumerate(t):
        for j, s_value in enumerate(changepoints):
            A[i][j] = 1 if t_value >= s_value else 0 
    # Cálculo vector gamma.
    gamma = -changepoints * delta
    # Cálculo "y real" e "y simulado", o con ruido.
    y_real = (k + np.dot(A, delta)) * t + (m + np.dot(A, gamma))
    y_sim =  y_real + np.random.randn(t_nr).reshape(-1,1) * sigma_error
    return y_sim, y_real

def linsplit(X, Y, gap):
    """
    Finds the best split point for a piecewise-linear fit of X vs Y.

    X, Y: numpy column vectors (nx1)
    gap:  minimum distance from edges

    Returns:
        s_est (float): estimated split index
        MSE   (array): [actual MSE, estimated MSE] with two columns
    """
    if X.shape[0]<=2*gap+1:
        return -1, None
    Xs = (X-X.min())/(X.max()-X.min())
    MSE_D2_est, beta_left_est, beta_right_est = mse_d2_est(Xs, Y, gap)
    idx = int(np.argmin(MSE_D2_est))
    s_opt = Xs[gap:-gap][idx][0]
    b1_est = beta_left_est[idx][0]
    b2_est = beta_right_est[idx][0]
    MSE_D2 = mse_d2(Xs[gap:-gap][:-1], s_opt, b1_est, b2_est)
    s_est = gap+idx
    MSE = np.concat((MSE_D2, MSE_D2_est), axis = 1)
    return s_est, MSE

def linsplit_iter(X, Y, tol_mse, n_max, wlen_signature, wlen_neighbor, tol_neighbor, tol_signature):
    """
    Iteratively scans long vectors X,Y and records good split points.

    tol_mse: min Pearson r² between empirical and theoretical MSE
    n_max:   max chunk size to process at once
    wlen_signature: half-window for signature search
    wlen_neighbor:  half-window to refine min-MSE signature
    tol_neighbor:   % of neighbors above the candidate minimum
    tol_signature:  threshold for signature detection (almost fixed)

    Returns:
        dict: {global_index: {'metric': r², 'gap': gap_used}}
    """
    last_idx = 0
    results = {}
    while last_idx<X.shape[0]:
        n_act = len(X[last_idx:])
        if n_act > n_max:
            X_act = X[last_idx:last_idx+n_max].copy()
            Y_act = Y[last_idx:last_idx+n_max].copy()
            n_act = n_max
        else:
            X_act = X[last_idx:].copy()
            Y_act = Y[last_idx:].copy()
        # next S candidate and end of signature 
        s_cand, eos = get_first_signature_pos(X_act, Y_act, wlen_signature, wlen_neighbor, tol_neighbor, tol_signature)
        if s_cand == -1:
            # Si no encuentra nada, se salta todo el bloque hasta el final.
            last_idx += n_act
            continue
        # recordar que epsilon tiene un valor fijo por defecto
        gap = get_gap(X_act[:s_cand], Y_act[:s_cand], epsilon = 0.05)
        # Acá estamos en un escenario donde se detecto the signature
        s_est, MSE = linsplit(X_act[:eos], Y_act[:eos], gap)
        if s_est == -1:
            # Aquí se guarda candidato y se recorre hasta el punto posterior a el.
            # candidate[last_idx+s_cand]=0
            last_idx += s_cand # o += gap
            continue
        # Metric for MSE_D2 
        r2_mse2 = coef_pearson(MSE[:,0], MSE[:,1])
        if r2_mse2>=tol_mse:
            last_idx += s_est
            results[last_idx] = {}
            results[last_idx]['metric'] = r2_mse2
            results[last_idx]['gap'] = gap
        else:
            last_idx += gap # Este valor es crítico, al cambiarlo puede cambiar el error total.
    return results

def linsplit_re(X, Y, results, tol_mse):
    """
    Re-runs linsplit inside every segment from `results` to find extra splits.

    X, Y: full data vectors
    results: dict of split points from previous pass
    tol_mse: min r² to accept a new split

    Returns:
        dict: updated split points (original ones kept)
    """
    gap_dict = {key: value['gap'] for key, value in results.items()}
    S_list = [0] + list(results.keys()) + [len(X)]
    S_list.sort()
    nr_chp = len(results)
    results_copy = {k:{sk:sv for sk, sv in v.items()} for k, v in results.items()}
    for idx in range(nr_chp+1):
        inf_idx, sup_idx = S_list[idx], S_list[idx+1]
        gap = max(gap_dict.get(inf_idx, 0), gap_dict.get(sup_idx, 0))
        s_est, MSE = linsplit(X[inf_idx:sup_idx], Y[inf_idx:sup_idx], gap)
        if s_est == -1:
            continue
        r2_mse2 = coef_pearson(MSE[:,0], MSE[:,1])
        if r2_mse2>=tol_mse:
            results_copy[inf_idx+s_est] = {}
            results_copy[inf_idx+s_est]['metric'] = r2_mse2
            results_copy[inf_idx+s_est]['gap'] = gap
    return results_copy

coef_pearson = lambda X, Y: np.corrcoef(X.ravel(), Y.ravel())[0][1]

