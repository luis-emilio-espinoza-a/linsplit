from linsplit import *
import numpy as np
import os

def rnd_slope_changes(n, b, cap):
    data = []
    for i in range(n):
        tmp_rnd = 0
        while abs(tmp_rnd)<=cap:
            tmp_rnd = np.random.laplace(0, b)
        data.append(tmp_rnd)
    return np.array(data, dtype=np.float64)

def rnd_s(rate, sup, cap):
    if rate>=sup:
        return None
    S = []
    S_cum = 0
    while S_cum < sup:
        S.append(S_cum)
        tmp_rnd = 0
        while tmp_rnd<cap or sup<tmp_rnd: # no cap, no empty.
            tmp_rnd = np.random.exponential(rate)
        S_cum += tmp_rnd
    return np.array(S[1:], dtype=np.float64)

def sim_xy(x_len, x_scale, alpha, s_rate, s_cap, beta_b, beta_cap):
    # x_len: number observations
    # x_scale: number observations per unit
    # alpha: intercept
    # s_rate: changepoint occurrence rate
    # s_cap: changepoint min. interval between two changepoint
    # beta_b: scale of slope changes
    # beta_cap: min. slope change

    while True:
        S = rnd_s(s_rate, x_len*x_scale, s_rate)
        if len(S)==7:
            break

    X = np.linspace(0, x_len*x_scale, x_len+1)
    Y = np.zeros_like(X)
    s_len = len(S)


    beta_slopes = rnd_slope_changes(s_len+1, beta_b, beta_cap)
    betas = np.zeros(s_len+1)
    alphas = np.zeros(s_len+1)

    betas[0] = beta_slopes[0]
    for i in range(s_len):
        betas[i+1] = betas[i] + beta_slopes[i]

    betas_diff = np.diff(betas, axis=0)
    alphas[0] = alpha
    alphas[1:] = (alpha - np.cumsum(betas_diff*S))

    parts = np.concat(([0], S, [X[-1]]))
    for i in range(s_len+1):
        part_inf, part_sup = parts[i], parts[i+1]
        if i < s_len:
            mask = np.logical_and(X>=part_inf,X<part_sup).ravel()
        else:
            mask = (X>=part_inf).ravel()
        Y[mask] = alphas[i]+betas[i]*X[mask]

    return X, Y, S, betas

def sim1(rewrite = False):

    sim_parameters = {
        'x_len': 10000, # number observations
        'x_scale': 1/100, # number observations per unit
        'alpha': 0, # intercept
        's_rate': 8, # changepoint occurrence rate
        's_cap': 5, # changepoint min. interval between two changepoint 
        'beta_b': 0.5, # scale of slope changes
        'beta_cap': 0.5, # min. slope change
        }

    params_linsplit_iter = {
        'wlen_signature': 100,
        'wlen_neighbor': 100,
        'tol_neighbor': 0.9,
        'tol_signature': 1,
        'tol_mse': 0.8,
        'n_max': 10000
        }

    sigma = 1

    if os.path.exists('./paper_data/sim1.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim1.npz')
        X = data_sim['X']
        Y = data_sim['Y']
        YS = data_sim['YS']
        S_est = data_sim['S_est']
        S_real = data_sim['S_real']
        betas = data_sim['betas']
    else:
        X, Y, S_real, betas = sim_xy(**sim_parameters)
        n = X.shape[0]
        X = X.reshape(n,1)
        Y = Y.reshape(n,1)
        Y = Y + np.random.normal(0,sigma,(n,1))
        results = linsplit_iter(X, Y, **params_linsplit_iter)
        YS = [Y[i][0] for i in results.keys()]
        S_est = [X[i][0] for i in results.keys()]
        S_real = list(S_real)
        np.savez('./paper_data/sim1.npz', X=X, Y=Y, YS = YS, S_est=S_est, S_real=S_real, betas=betas)

    return X, Y, YS, S_est, S_real, betas

def sim2(rewrite = False):

    params_linsplit_iter = {
        'wlen_signature': 70,
        'wlen_neighbor': 70,
        'tol_neighbor': 0.7,
        'tol_signature': 1,
        'tol_mse': 0.8,
        'n_max': 10000
        }

    tol_mse = params_linsplit_iter['tol_mse']

    if os.path.exists('./paper_data/sim2.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim2.npz')
        Xs = data_sim['Xs']
        S = data_sim['S']
    else:
        s_list = []
        S = [1, 3, 4, 8]
        for i in range(1000):
            # Simulación
            n = 1000
            X = np.linspace(0,10,n).reshape(-1,1)
            X = (np.random.rand(n)*10).reshape(-1,1)
            X = np.sort(X, axis=0)
            rates = np.array([1, 0.5, -0.3, -1, -0.2]).reshape(-1,1)
            changepoints = np.array(S).reshape(-1,1)
            m = 0
            sigma_error = 0.1
            Y, _ = piecewise_regression_sim(X, rates, changepoints, m, sigma_error)

            results = linsplit_iter(X, Y, **params_linsplit_iter).copy()
            if len(results)>1:
                results = linsplit_re(X, Y, results, tol_mse).copy()

            s_list.extend(list(results.keys()))
        Xs = X[s_list].ravel()
        np.savez('./paper_data/sim2.npz', Xs = Xs, S=S)

    return Xs, S

def sim3(rewrite = False):

    params_linsplit_iter = {'wlen_signature': 50,
    'wlen_neighbor': 90,
    'tol_neighbor': 0.7,
    'tol_signature': 1,
    'tol_mse': 0.6,
    'n_max': 10000
    }

    tol_mse = params_linsplit_iter['tol_mse']

    if os.path.exists('./paper_data/sim3.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim3.npz')
        Xs = data_sim['Xs']
        S = data_sim['S']
    else:
        s_list = []
        S = [1, 3, 4, 8]
        for i in range(1000):
            # Simulación
            n = 1000
            X = np.linspace(0,10,n).reshape(-1,1)
            X = (np.random.rand(n)*10).reshape(-1,1)
            X = np.sort(X, axis=0)
            rates = np.array([1, 0.5, -0.3, -1, -0.2]).reshape(-1,1)
            changepoints = np.array(S).reshape(-1,1)
            m = 0
            sigma_error = 0.3
            Y, _ = piecewise_regression_sim(X, rates, changepoints, m, sigma_error)

            results = linsplit_iter(X, Y, **params_linsplit_iter).copy()
            if len(results)>1:
                results = linsplit_re(X, Y, results, tol_mse).copy()

            s_list.extend(list(results.keys()))

        Xs = X[s_list].ravel()
        np.savez('./paper_data/sim3.npz', Xs = Xs, S=S)

    return Xs, S

def sim4():

    signature = lambda S, T: 2*S**3/T**3-3*S**2/T**2

    nro_sims = 30
    R = [1, -1, 1, -1, 1]
    S = [2, 4, 6, 8]
    m = 0
    n = 1000
    sigma_error = 0.75
    beta_empirical_list = []
    X = np.linspace(0,10,n).reshape(-1,1)
    rates = np.array(R).reshape(-1,1)
    changepoints = np.array(S).reshape(-1,1)

    Y_list = []
    for i in range(nro_sims):
        Y, _ = piecewise_regression_sim(X, rates, changepoints, m, sigma_error)
        Y_list.append(Y)
        beta = beta_vect(X, Y)
        if i == 0:
            gap = get_gap(X[X<=S[0]].reshape(-1,1), Y[X<=S[0]].reshape(-1,1), 0.01)
        beta_empirical_list.append(beta)

    beta_empirical = np.concat(beta_empirical_list, axis=1)

    # Theoretical values firts part.
    S_theo = changepoints[0][0]
    b1_theo = rates[0][0]
    b2_theo = rates[1][0]
    beta_signature = np.zeros_like(X)
    beta_signature[X<=S_theo] = b1_theo
    beta_signature[X>S_theo] = (b2_theo-b1_theo)*signature(S_theo,X[X>S_theo])+b2_theo

    X_1 = X.copy()
    beta_empirical_1 = beta_empirical.copy()
    beta_signature_1 = beta_signature.copy()
    S_1 = S.copy()
    b1_theo_1 = b1_theo

    # Second Part.

    X2 = X[X>S_theo].reshape(-1,1)
    X2_min = X2.min()

    X2 = X2-X2_min

    # Theoretical values
    S_theo = changepoints[1][0]-X2_min
    b1_theo = rates[1][0]
    b2_theo = rates[2][0]

    beta_empirical_list = []
    for i in range(nro_sims):
        Y2 = Y_list[i][X>S_theo].reshape(-1,1)
        beta = beta_vect(X2, Y2)
        beta_empirical_list.append(beta)

    beta_empirical_2 = np.concat(beta_empirical_list, axis=1)

    # Empirical values
    beta_signature = np.zeros_like(X2)
    beta_signature[X2<=S_theo] = b1_theo
    beta_signature[X2>S_theo] = (b2_theo-b1_theo)*signature(S_theo,X2[X2>S_theo])+b2_theo

    X_2 = X2.copy()+X2_min
    beta_signature_2 = beta_signature.copy()
    S_2 = S[1:].copy()
    b1_theo_2 = b1_theo
    return X_1, beta_empirical_1, beta_signature_1, S_1, b1_theo_1, X_2, beta_empirical_2, beta_signature_2, S_2, b1_theo_2, gap

def sim5(sigma, nro_sims = 10):
    s = 0.6
    n = 1000
    a = 0.7
    b1 = 1
    b2 = -1

    MSE_est_sim = []
    beta_left_sim = []
    beta_right_sim = []
    s_opt_list = []

    for i in range(nro_sims):

        X = np.linspace(0,1,n).reshape(n,1)
        Y = np.zeros_like(X)
        Y[X<s] = a + b1 * X[X<s]
        Y[X>=s] = a + b2 * X[X>=s] - s * (b2 - b1)
        Y = Y + np.random.normal(0,sigma, (n,1))
        if i == 0:
            gap = get_gap(X[X<s].reshape(-1,1), Y[X<s].reshape(-1,1), epsilon = 0.01)

        MSE_D2_est, beta_left_est, beta_right_est = mse_d2_est(X, Y, gap)
        MSE_est_sim.append(MSE_D2_est)
        idx = int(np.argmin(MSE_D2_est))
        s_opt = X[gap:-gap][idx][0]

        beta_left_sim.append(beta_left_est[1:])
        beta_right_sim.append(beta_right_est[:-1])
        s_opt_list.append(s_opt)

    MSE_est_sim = np.concat(MSE_est_sim, axis = 1)
    beta_left_sim = np.concat(beta_left_sim, axis = 1)
    beta_right_sim = np.concat(beta_right_sim, axis = 1)

    X = X[gap:-gap-1]

    MSE_D2_theoretical = mse_d2(X, s, b1, b2)
    beta_left_theoretical = (b1*s**2*(3*X-2*s)+b2*(X-s)**2*(X+2*s))/X**3
    beta_left_theoretical[X<s] = b1

    beta_right_theoretical = (b1*(s-X)**2*(3-2*s-X)+b2*(1-s)**2*(1+2*s-3*X))/(1-X)**3
    beta_right_theoretical[X>s] = b2

    return X, MSE_est_sim, MSE_D2_theoretical, beta_left_sim, beta_left_theoretical, beta_right_sim, beta_right_theoretical, s_opt_list

def sim6(rewrite = False):

    if os.path.exists('./paper_data/sim6.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim6.npz')
        X = data_sim['X']
        Y = data_sim['Y']
        D2_sim = data_sim['D2_sim']
        D2_theo = data_sim['D2_theo']
        gap = data_sim['gap']
        s = data_sim['s']
    else:
        s = 0.6
        n = 1000
        a = 0.7
        b1 = 1
        b2 = -1
        sigma = 0.05

        nro_sims = 10

        bl_sim = []
        br_sim = []
        S = []

        for i in range(nro_sims):

            X = np.linspace(0,1,n).reshape(n,1)
            Y = np.zeros_like(X)
            Y[X<s] = a + b1 * X[X<s]
            Y[X>=s] = a + b2 * X[X>=s] - s * (b2 - b1)
            Y = Y + np.random.normal(0,sigma, (n,1))
            if i == 0:
                gap = get_gap(X[X<s].reshape(-1,1), Y[X<s].reshape(-1,1), epsilon = 0.01)

            MSE_D2_est, _, _ = mse_d2_est(X, Y, gap)
            idx = int(np.argmin(MSE_D2_est))
            s_opt = X[gap:-gap][idx][0]

            bl = beta_vect(X, Y, gap_left = 2, gap_right = 2)
            br = beta_vect(X[::-1], Y[::-1], gap_left = 2, gap_right = 2)
            br = br[::-1]

            bl_sim.append(bl)
            br_sim.append(br)
            S.append(s_opt)

        bl_sim = np.concat(bl_sim, axis = 1)
        br_sim = np.concat(br_sim, axis = 1)

        bl_theo = (b1*s**2*(3*X-2*s)+b2*(X-s)**2*(X+2*s))/X**3
        bl_theo[X<s] = b1

        br_theo = (b1*(s-X)**2*(3-2*s-X)+b2*(1-s)**2*(1+2*s-3*X))/(1-X)**3
        br_theo[X>s] = b2

        # Deleting data null
        X = X[2:-2].copy()
        Y = Y[2:-2].copy()
        bl_sim = bl_sim[2:-2,:].copy()
        br_sim = br_sim[2:-2,:].copy()
        bl_theo = bl_theo[2:-2,:].copy()
        br_theo = br_theo[2:-2,:].copy()
        gap -= 2
        D2_sim = (bl_sim-br_sim)**2
        D2_theo = (bl_theo-br_theo)**2

        np.savez('./paper_data/sim6.npz', X = X, Y = Y, D2_sim = D2_sim, D2_theo = D2_theo, gap = gap, s = s)

    return X, Y, D2_sim, D2_theo, gap, s

def sim7(sigma):
    S = [1, 3, 4, 8]
    n = 1000
    X = (np.random.rand(n)*10).reshape(-1,1)
    X = np.sort(X, axis=0)
    rates = np.array([1, 0.5, -0.3, -1, -0.2]).reshape(-1,1)
    changepoints = np.array(S).reshape(-1,1)
    m = 0
    Y, _ = piecewise_regression_sim(X, rates, changepoints, m, sigma)
    return X, Y, S

def sim8(sigma, rewrite = False):

    if os.path.exists('./paper_data/sim8.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim8.npz')
        X = data_sim['X']
        Y = data_sim['Y']
        D2 = data_sim['D2']
    else:
        s = 0.6
        n = 1000
        a = 0.7
        b1 = 1
        b2 = -1
        X = np.linspace(0,1,n).reshape(n,1)
        Y = np.zeros_like(X)
        Y[X<s] = a + b1 * X[X<s]
        Y[X>=s] = a + b2 * X[X>=s] - s * (b2 - b1)
        Y = Y + np.random.normal(0,sigma, (n,1))
        bl = beta_vect(X, Y, gap_left = 2, gap_right = 2)
        br = beta_vect(X[::-1], Y[::-1], gap_left = 2, gap_right = 2)
        br = br[::-1]
        X = X[2:-2].copy()
        Y = Y[2:-2].copy()
        bl = bl[2:-2].copy()
        br = br[2:-2].copy()
        D2 = (bl-br)**2
        np.savez('./paper_data/sim8.npz', X = X, Y = Y, D2 = D2)

    return X, Y, D2

def sim9(n, nro_sims, rewrite = False):
    s = 0.6
    a = 0.7
    b1 = 1
    b2 = -1
    sigma = 0.25
    S = []
    if os.path.exists('./paper_data/sim9.npz') and not rewrite:
        data_sim = np.load('./paper_data/sim9.npz')
        X = data_sim['X']
        S = data_sim['S']
    else:
        for i in range(nro_sims):

            X = np.linspace(0,1,n).reshape(n,1)
            Y = np.zeros_like(X)
            Y[X<s] = a + b1 * X[X<s]
            Y[X>=s] = a + b2 * X[X>=s] - s * (b2 - b1)
            Y = Y + np.random.normal(0,sigma, (n,1))

            gap = get_gap(X, Y, 0.01)
            s_est, _ = linsplit(X, Y, gap)
            S.append(float(X[s_est][0]))
        np.savez('./paper_data/sim9.npz', X = X, S = S)
    return X, np.array(S).reshape(nro_sims, 1)

# from time import time
# ini = time()
# n = 1000
# nro_sims = 1000
# X, S = sim9(n, nro_sims)
# print(time()-ini)
# time values by n:
#   1000 = 0.007
#  10000 = 0.056
# 100000 = 0.688
#1000000 = 6.096

# n = 100000
# T = np.linspace(1/n,(n-1)/n,n-1).reshape(n-1,1)
# s = 0.5
# b1 = 1
# b2 = -1
# from time import time
# ini = time()
# for _ in range(10):
#     _ = mse_d2(T, s, b1, b2)
# total_time = time()-ini
# print(total_time/10)
