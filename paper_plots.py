import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from paper_simulations import *
import numpy as np

# Configuración de gráficos.
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['grid.alpha'] = 0.3
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
rcParams['font.size'] = 11
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10

S_lines_params = {
    'color': 'black',
    'linestyle': '--',
    'linewidth': 1.2,
    'alpha': 0.7,

}

gap_lines_params = {
    'color': 'black',
    'linestyle': '-.',
    'linewidth': 1.2
}

size = (10, 6)

color_main = 'dimgray'
color_second = 'silver'

#################### many_changepoints

X, Y, YS, S, S_real, betas = sim1()

# El siguiente cálculo no cambia los S estimados, sólo a qué altura con respecto al eje Y
# se muestra el punto de cambio en el gráfico ((ax.scatter).
diff_x = np.diff(X.ravel()).mean()
epsilon = 10
YS  = [Y[np.logical_and(X>=s-diff_x*epsilon, X<=s+diff_x*epsilon)].mean() for s in S]

fig, ax = plt.subplots(figsize=size)

ax.plot(X, Y, color=color_main, linewidth=0.5, alpha=0.7, label='Observed data')

# Único scatter.
ax.scatter(S, YS, 
           s=80, 
           c=color_second, 
           marker='o', 
           edgecolors='white',
           linewidth=1.5, 
           label='Detected changepoints',
           zorder=5)

for i, s in enumerate(S_real):
    ax.axvline(x=s,
    **S_lines_params,
    label='Truth changepoints' if i == 0 else None)

for i, (s, y) in enumerate(zip(S, YS)):
    ax.annotate(f'S{i+1}', 
                xy=(s, y), 
                xytext=(4, 12 if i > 0 else -12), 
                textcoords='offset points',
                fontsize=9,
                color='black',
                fontweight='bold')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Changepoints detection', pad=20)

ax.set_xlim(0, 100)
ax.set_ylim(Y.min() - 5, Y.max() + 5)

ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.9)

plt.tight_layout()

plt.savefig('./paper_plots/pdfs/many_changepoints.pdf', format='pdf', bbox_inches='tight')
plt.savefig('./paper_plots/pngs/many_changepoints.png', format='png', dpi=300, bbox_inches='tight')
plt.clf()

#################### GRAFICO 2
Xs1, S1 = sim2() # sigma = 0.1
Xs2, S2 = sim3() # sigma = 0.3


cases = [
    [Xs1, S1],
    [Xs2, S2]
]

for i_case, case in enumerate(cases):
    Xs, S = case

    fig, ax = plt.subplots(figsize=size)

    N, bins, patches = ax.hist(
        Xs,
        bins=100,
        color=color_main,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5
    )

    # colorear las barras según proximidad a los valores verdaderos
    for b0, b1, p in zip(bins[:-1], bins[1:], patches):
        center = (b0 + b1) / 2
        if any(abs(center - t) < 0.3 for t in S):
            p.set_facecolor(color_second)
            p.set_alpha(0.9)

    for t in S:
        ax.axvline(t,
        **S_lines_params
        )

    # ajustes cosméticos
    ax.set_xlabel('X')
    ax.set_ylabel('Frequency')
    if i_case == 0:
        ax.set_title(r'Distribution of Detected Changepoints (No. of sims = 1,000, $\sigma$=0.1)', pad=20)
    else:
        ax.set_title(r'Distribution of Detected Changepoints (No. of sims = 1,000 sims, $\sigma$=0.3)', pad=20)

    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0, 11))   # 0,1,2,...,10
    ax.set_xticklabels([str(i) for i in range(11)])

    ax.legend(handles=[plt.Line2D([0], [0], color=color_second, lw=4, label='Detected changepoints'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Truth changepoints')],
            loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    if i_case == 0:
        plt.savefig('./paper_plots/pdfs/distribution_changepoints_01.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('./paper_plots/pngs/distribution_changepoints_01.png', format='png', dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.savefig('./paper_plots/pdfs/distribution_changepoints_03.pdf', format='pdf', bbox_inches='tight')
        plt.savefig('./paper_plots/pngs/distribution_changepoints_03.png', format='png', dpi=300, bbox_inches='tight')
        plt.clf()


# THE SIGNATURE: beta empirical (b_e) vs beta signature (b_s)

X_1, b_e_1, b_s_1, S_1, b1_theo_1,\
     X_2, b_e_2, b_s_2, S_2, b1_theo_2, gap = sim4()

cases = [
    [X_1, b_e_1, b_s_1, S_1, b1_theo_1],
    [X_2, b_e_2, b_s_2, S_2, b1_theo_2]
]

for i_case, case in enumerate(cases):
    X, b_e, b_s, S, b1_theo = case

    fig, ax = plt.subplots(figsize=size)

    nro_sims = b_e.shape[1]

    for i in range(nro_sims):
        ax.plot(X[:gap], b_e[:, i][:gap],
                color=color_second,
                linewidth=1,
                alpha=0.3)

    for i in range(nro_sims):
        ax.plot(X[gap:], b_e[:, i][gap:],
                color=color_second,
                linewidth=1,
#                alpha=0.8,
                label=r'Empirical: $\hat{\beta}_{t}$' if i == 0 else None)

    ax.plot(X, b_s,
            color=color_main,
            linewidth=2,
#            alpha = 0.8,
            label=r'The Hermite Signature')

    for i, s in enumerate(S):
        ax.axvline(s,
        **S_lines_params,
        label='Changepoints' if i == 0 else None)

    ax.axvline(X[gap][0],
                **gap_lines_params,
                label='gap')

    if i_case == 0:
        ax.set_xticks(np.arange(0, 11,2))
        ax.set_xlim(0, 10)
    else:
        ax.set_xticks(np.arange(2, 11,2))
        ax.set_xlim(2, 10)
    ax.set_xlabel('X')
    ax.set_ylabel(r'$\beta_{t}$')
#    ax.set_title('Simulated Empirical vs. Theoretical Betas: The Hermite Signature', pad=20)
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

    factor = 0.15
    sig_min, sig_max = b_s.min(), b_s.max()
    diff_from_b1 = max(b1_theo-sig_min, sig_max-b1_theo)
    lower_y = b1_theo-diff_from_b1*(1+factor)
    upper_y = b1_theo+diff_from_b1*(1+factor)
    plt.ylim(lower_y, upper_y)

    plt.tight_layout()
    plt.savefig(f'./paper_plots/pdfs/the_signature_{i_case}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'./paper_plots/pngs/the_signature_{i_case}.png', format='png', dpi=300, bbox_inches='tight')
    plt.clf()

# THE MSE, beta_left, beta_right

X1, MSE_e, MSE_t, _, _, _, _, _ = sim5(0.05)
X2, _, _, beta_left_e, beta_left_t, beta_right_e, beta_right_t, S = sim5(0.2)
X3, MSE_e_3, MSE_t_3, _, _, _, _, _ = sim5(0.25, nro_sims = 30)

cases = [
    [X1, MSE_e, MSE_t],
    [X2, beta_left_e, beta_left_t],
    [X2, beta_right_e, beta_right_t],
    [X3, MSE_e_3, MSE_t_3]
]

filenames = ['MSE', 'beta_left', 'beta_right', 'MSE_chaotic']

Y_label = ['EMC', r'$\beta_{\text{left}}$', r'$\beta_{\text{right}}$', 'EMC']

Y_lim = [
    [0, max(MSE_t)*1.25],
    [beta_left_t[0][0]-1.5,beta_left_t[0][0]+1.5],
    [beta_right_t[-1][0]-1.5, beta_right_t[-1][0]+1.5],
    [0, max(MSE_t_3)*1.25]
]

for i_case, case in enumerate(cases):
    X, sim, theo = case
    y_inf, y_sup = Y_lim[i_case]

    fig, ax = plt.subplots(figsize=size)

    nro_sims = sim.shape[1]

    for i in range(nro_sims):
        ax.plot(X, sim[:,i],
                color=color_second,
                linewidth=1,
#                alpha=0.3,
                label='Empirical Value' if i == 0 else None)

    ax.plot(X, theo,
            color=color_main,
            linewidth=2,
#            alpha = 0.8,
            label='Theoretical Value')

    ax.axvline(X[0][0], 
            **gap_lines_params)

    ax.axvline(X[-1][0],
            **gap_lines_params,
                label='gap')

    ax.axvline(0.6,
            **S_lines_params,
            label='Changepoint')

    ax.set_xlabel('X')
    ax.set_ylabel(Y_label[i_case])
#    ax.set_title(titles[i_case], pad=20)
    ax.set_xlim(0, 1)

    ax.set_ylim(y_inf, y_sup)

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(f'./paper_plots/pdfs/{filenames[i_case]}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'./paper_plots/pngs/{filenames[i_case]}.png', format='png', dpi=300, bbox_inches='tight')
    plt.clf()

# D2 theoretical empirical gap

X, Y, D2_sim, D2_theo, gap, s = sim6(True)

fig, ax = plt.subplots(figsize=size)

nro_sims = D2_sim.shape[1]

for i in range(nro_sims):
    ax.plot(X[:gap], D2_sim[:gap][:,i],
            color=color_second,
            linewidth=1,
            alpha=0.3)
    ax.plot(X[-gap:], D2_sim[-gap:][:,i],
            color=color_second,
            linewidth=1,
            alpha=0.3)
    ax.plot(X[gap:-gap], D2_sim[gap:-gap][:,i],
            color=color_second,
            linewidth=1,
#            alpha=0.8,
            label='Empirical Value' if i == 0 else None)

ax.plot(X, D2_theo,
        color=color_main,
        linewidth=2,
#        alpha = 0.8,
        label='Theoretical Value')

ax.axvline(X[gap][0],
        **gap_lines_params)

ax.axvline(X[-gap][0],
        **gap_lines_params,
            label='gap')

ax.axvline(s,
        **S_lines_params,
        label='Changepoint')

ax.set_xlabel('X')
ax.set_ylabel(r'$D_t$')
#ax.set_title('blablabla', pad=20)
ax.set_xlim(0, 1)

ax.set_ylim(0, np.max(D2_theo)*1.15)

ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()

plt.tight_layout()
plt.savefig(f'./paper_plots/pdfs/D2.pdf', format='pdf', bbox_inches='tight')
plt.savefig(f'./paper_plots/pngs/D2.png', format='png', dpi=300, bbox_inches='tight')
plt.clf()

## X vs Y

X1, Y1, S1 = sim7(0.1)
X2, Y2, S2 = sim7(0.3)

cases = [
    [X1, Y1, S1],
    [X2, Y2, S2]
]

for i_case, case in enumerate(cases):
    X, Y, S = case
    fig, ax = plt.subplots(figsize=size)
    for i, s in enumerate(S):
        ax.axvline(s,
        **S_lines_params,
        label='Changepoints' if i == 0 else None)
    ax.set_xticks(np.arange(0, 11 ,2))
    ax.set_xlim(0, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.plot(X, Y,
        color=color_main,
        alpha = 0.8
    )

    plt.tight_layout()
    plt.savefig(f'./paper_plots/pdfs/XY_{str(i_case)}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'./paper_plots/pngs/XY_{str(i_case)}.png', format='png', dpi=300, bbox_inches='tight')
    plt.clf()

## D empírica

X, Y, D2 = sim8(0.25)

y_lim1 = Y.max()*1.25
y_lim2 = D2[np.logical_and(X<=0.7, X>=0.6)].max()*4

D2.max()

cases = [
    [X, Y, y_lim1, 'Y', 'XY_D2'],
    [X, D2, y_lim2, r'$\hat{D}(x_i)$', 'D2_D2']
]

for i, case in enumerate(cases):
    X, Y, y_lim, ylabel, title = case

    fig, ax = plt.subplots(figsize=size)
    ax.set_xticks(np.arange(0, 1.1 ,0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_lim)

    ax.axvline(0.6,
    **S_lines_params,
    label='Changepoint')


    ax.set_xlabel('X')
    ax.set_ylabel(ylabel)
    if i == 0:
        ax.plot(X, Y, '.',
            color=color_main,
            alpha = 0.8
        )
    else:
        ax.plot(X, Y,
            color=color_main,
            alpha = 0.8
        )

    ax.legend(loc='upper center', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'./paper_plots/pdfs/{title}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'./paper_plots/pngs/{title}.png', format='png', dpi=300, bbox_inches='tight')
    plt.clf()
# 

#################### GRAFICO frecuencias 2
X, S = sim9(1000, 10000)

fig, ax = plt.subplots(figsize=size)
S.mean()
S.std()
[np.percentile(S, 2.5), np.percentile(S, 97.5)]
[np.percentile(S, 10), np.percentile(S, 90)]

N, bins, patches = ax.hist(
    S,
    bins=100,
    color=color_main,
    alpha=0.7,
    edgecolor='white',
    linewidth=0.5
)

# colorear las barras según proximidad a los valores verdaderos
for b0, b1, p in zip(bins[:-1], bins[1:], patches):
    center = (b0 + b1) / 2
    if any(abs(center - t) < 0.3 for t in S):
        p.set_facecolor(color_second)
        p.set_alpha(0.9)

ax.axvline(0.6, **S_lines_params)

# ajustes cosméticos
ax.set_xlabel('X')
ax.set_ylabel('Frequency')
ax.set_title(r'Distribution of Detected Changepoints (No. of sims =10,000)', pad=20)

ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.2))

ax.legend(handles=[plt.Line2D([0], [0], color=color_second, lw=4, label='Detected changepoint'),
                plt.Line2D([0], [0], color='black', linestyle='--', label='Truth changepoint')],
        loc='best', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('./paper_plots/pdfs/hist_main.pdf', format='pdf', bbox_inches='tight')
plt.savefig('./paper_plots/pngs/hist_main.png', format='png', dpi=300, bbox_inches='tight')
plt.clf()
