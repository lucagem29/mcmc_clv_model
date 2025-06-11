import math
import numpy as np, math
from numba import njit, prange


@njit
def p_alive_jit(lam, mu, T, t_x):
    return 1.0 / (1.0 + (mu / (lam + mu)) * (np.exp((lam + mu) * (T - t_x)) - 1.0))


@njit
def trunc_exp_jit(rate, lower, upper, u):
    """Draw from Exp(rate) truncated to (lower, upper) given U~Unif(0,1)."""
    return -math.log(math.exp(-rate * upper) +
                     u * (math.exp(-rate * lower) - math.exp(-rate * upper))) / rate


@njit
def log_complete_lik_jit(theta, x, t_x, T, z, y, d, beta, Gamma, log_det_Gamma, Ginv00, Ginv01, Ginv11):
    lam, mu = math.exp(theta[0]), math.exp(theta[1])

    # likelihood terms
    ll = 0.0
    if x > 0:
        ll += x*math.log(lam) + (x-1.0)*math.log(t_x) - math.lgamma(x) - lam*t_x
    if z == 1:
        ll += -(lam + mu) * (T - t_x)
    else:
        ll += math.log(mu) - (lam + mu) * (y - t_x) - (lam + mu) * (T - y)

    # prior: MVN(theta | beta^T d, Gamma)
    m0 = d[0]*beta[0, 0] + d[1]*beta[1, 0]
    m1 = d[0]*beta[0, 1] + d[1]*beta[1, 1]
    diff0 = theta[0] - m0
    diff1 = theta[1] - m1
    quad = Ginv00*diff0*diff0 - 2*Ginv01*diff0*diff1 + Ginv11*diff1*diff1
    ll += -0.5*(quad + log_det_Gamma + 2*math.log(2*math.pi))

    return ll

@njit
def log_complete_lik_numba(theta0, theta1, x, tx, T, z, y,
                           m0, m1, Ginv00, Ginv01, Ginv11, logdetG):
    lam = math.exp(theta0)
    mu  = math.exp(theta1)
    # Eq. 5 likelihood
    ll = 0.0
    if x > 0:
        ll += x*math.log(lam) + (x-1.0)*math.log(tx) - math.lgamma(x) - lam*tx
    if z == 1:
        ll += -(lam+mu)*(T-tx)
    else:
        ll += math.log(mu) - (lam+mu)*(y-tx) - (lam+mu)*(T-y)
    # MVN prior
    d0 = theta0 - m0
    d1 = theta1 - m1
    quad = Ginv00*d0*d0 - 2*Ginv01*d0*d1 + Ginv11*d1*d1
    ll += -0.5*(quad + logdetG + 2*math.log(2*math.pi))
    return ll

@njit(parallel=True)
def update_customers(theta, z, y, beta, Ginv, logdetG,
                     x, t_x, T, D, proposal_scale, rng_norm, rng_uni):
    """
    One Gibbs/Metropolis sweep over *all* customers, fully in Numba land.
    Arrays are all 1-D; shapes are documented at the call-site.
    Returns the number of Metropolis accepts.
    """
    N = theta.shape[0]
    accept = 0
    for i in prange(N):

        # 1. unpack
        th0, th1 = theta[i, 0], theta[i, 1]
        lam = math.exp(th0)
        mu  = math.exp(th1)

        # 2a. sample zᵢ
        p_alive = 1.0 / (1.0 + (mu/(lam+mu)) *
                         (math.exp((lam+mu)*(T[i]-t_x[i])) - 1.0))
        z[i] = 1 if rng_uni[i] < p_alive else 0

        # 2b. sample yᵢ if inactive
        if z[i] == 0:
            u = rng_uni[N+i]                       # use another stream
            lower = 0.0
            upper = T[i] - t_x[i]
            y[i] = -math.log(math.exp(-(lam+mu)*upper)
                             + u*(math.exp(-(lam+mu)*lower)
                                  - math.exp(-(lam+mu)*upper))) / (lam+mu)
        else:
            y[i] = math.nan

        # 2c. RW-Metropolis step on θᵢ
        prop0 = th0 + rng_norm[2*i]   * proposal_scale
        prop1 = th1 + rng_norm[2*i+1] * proposal_scale

        # compute log-lik current and proposal
        def _ll(th0_, th1_):
            lam_ = math.exp(th0_);  mu_ = math.exp(th1_)
            ll = 0.0
            if x[i] > 0:
                ll += (x[i]* math.log(lam_) + (x[i]-1)* math.log(t_x[i])
                       - math.lgamma(x[i]) - lam_*t_x[i])
            if z[i] == 1:
                ll += -(lam_+mu_)*(T[i]-t_x[i])
            else:
                ll += (math.log(mu_) - (lam_+mu_)*(y[i]-t_x[i])
                       - (lam_+mu_)*(T[i]-y[i]))
            # MVN prior
            m0 = D[i,0]*beta[0,0] + D[i,1]*beta[1,0]
            m1 = D[i,0]*beta[0,1] + D[i,1]*beta[1,1]
            d0 = th0_ - m0
            d1 = th1_ - m1
            quad = (Ginv[0,0]*d0*d0 - 2*Ginv[0,1]*d0*d1 + Ginv[1,1]*d1*d1)
            ll += -0.5*(quad + logdetG + math.log(4*math.pi*math.pi))
            return ll

        log_acc = _ll(prop0, prop1) - _ll(th0, th1)
        if math.log(rng_uni[2*N+i]) < log_acc:
            theta[i, 0] = prop0
            theta[i, 1] = prop1
            accept += 1

    return accept