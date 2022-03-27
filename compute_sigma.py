import numpy as np
from scipy import optimize
from scipy.stats import norm
from scipy.special import gammainc
from tqdm import tqdm


def compute_mu_poisson(T, sigma, n, batch_size):
    """Compute mu from Poisson subsampling."""
    return np.sqrt((np.exp(sigma ** (-2)) - 1) * T) * batch_size / n


def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
    """Compute epsilon from mu given delta via inverse dual."""
    return optimize.root_scalar(lambda x: delta_eps_mu(x, mu) - delta, bracket=[0, 700], method='brentq').root


# def compute_eps_poisson(epoch, noise_multi, n, batch_size, delta):
#     """Compute epsilon given delta from inverse dual of Poisson subsampling."""
#
#     return eps_from_mu(
#         compute_mu_poisson(epoch, noise_multi, n, batch_size), delta)

def sigma_from_eps(delta, epsilon, T, total_size, batch_size):  # function computing sigma based on delta and epsilon
    return optimize.root_scalar(
        lambda x: eps_from_mu(compute_mu_poisson(T, x, total_size, batch_size), delta) - epsilon, bracket=[3, 100],
        method='brentq').root


# sigma_from_eps(0.01, 10, T=6000, total_size=2, batch_size=2)
# when delta=0.01, if we want to reduce epsilon to 0.1, sigma need to be 739

# sigma_from_eps(0.01, 2, T=6000, total_size=2, batch_size=2)

# sigma_from_eps(0.01, 5, T=6000, total_size=100, batch_size=10)

def p_from_eps_delta(delta, epsilon, T, total_size, batch_size, tol_prob, tol_thr):
    return optimize.root_scalar(lambda x: gammainc(x / 2,
                                                   tol_thr ** 2 * batch_size ** 2 / sigma_from_eps(delta, epsilon, T,
                                                                                                   total_size,
                                                                                                   batch_size) ** 2 / 8) - tol_prob,
                                bracket=[1, 10 ** 9], method='brentq').root


def rho_from_p_eps_delta(delta, epsilon, T, total_size, batch_size, p, tol_thr):
    sigma = sigma_from_eps(delta, epsilon, T, total_size, batch_size)
    return gammainc(p / 2, tol_thr ** 2 * batch_size ** 2 / sigma ** 2 / 8)


def rho_from_p_sigma(sigma, p, T, batch_size, tol_thr):
    return gammainc(p / 2, T * tol_thr ** 2 * batch_size ** 2 / sigma ** 2 / 2)


def thr_from_p_sigma(sigma, p, T, batch_size, rho):
    return optimize.root_scalar(lambda x: gammainc(p / 2, T * x ** 2 * batch_size ** 2 / sigma ** 2 / 4) - rho,
                                bracket=[1e-16, 50],
                                method='brentq').root


def sigma_from_eps_delta(delta, epsilon, T, total_size, batch_size):
    mu = optimize.root_scalar(lambda x: delta_eps_mu(epsilon, x) - delta, bracket=[-100, 100], method='brentq').root
    return optimize.root_scalar(lambda x: compute_mu_poisson(T, x, total_size, batch_size) - mu, bracket=[0.1, 100],
                                method='brentq').root


if __name__ == "__main__":
    # p_upper = p_from_eps_delta(0.01, 5, 6000, 1000, 100, 0.2, 1.0)
    # print(p_upper)
    # rho_act = rho_from_p_eps_delta(0.01, 5, 6000, 1000, 100, 816, 2.0)
    # print(rho_act)
    # rho = rho_from_p_sigma(5.5, 229180, 25000, 100, 0.25)
    # print(rho)
    # thr = thr_from_p_sigma(1.34, 229180, 6000, 100, 0.05)
    # print(thr)
    # sigma = sigma_from_eps_delta(1e-2, 1.0, T=10000, total_size=64, batch_size=16)
    # print(sigma)
    #
    # print(thr_from_p_sigma(1.5, 778140, 10000, 42, 0.95))
    print(eps_from_mu(compute_mu_poisson(10000, 10, 64, 21), 1e-2))
    # sigmas = []
    # for eps in np.linspace(1, 10, num=100):
    #     sigmas.append(sigma_from_eps_delta(delta=1e-5, epsilon=eps, T=20000, total_size=60000, batch_size=256))
    # np.savetxt("sigmas.txt", sigmas)
    # print(sigmas)




    # DPRC
    # thrs = []
    # probs= []
    # for n in tqdm(range(1000, 50001, 20)):
    #     newt = []
    #     newp = []
    #     for T in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #         newt.append(thr_from_p_sigma(5, 2000, T, n * 0.1, 0.95))
    #         newp.append(rho_from_p_sigma(5, 2000, T, n * 0.1, 0.02))
    #     thrs.append(newt)
    #     probs.append(newp)
    # # print(thrs)
    # np.savetxt("thrs.txt", thrs)
    # np.savetxt("probs.txt", probs)

    # thrs = []
    # probs = []
    # for n in tqdm(range(1000, 50001, 20)):
    #     newt = []
    #     newp = []
    #     for p in np.linspace(0.01, 0.99, 15):
    #         newt.append(thr_from_p_sigma(5, 2000, 500, n * 0.1, p))
    #     for t in [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064]:
    #         newp.append(rho_from_p_sigma(5, 2000, 500, n * 0.1, t))
    #     thrs.append(newt)
    #     probs.append(newp)
    #
    # np.savetxt("thrs_a.txt", thrs)
    # np.savetxt("probs_a.txt", probs)
    #
    # thrs = []
    # probs = []
    # for n in tqdm(range(1000, 50001, 20)):
    #     newt = []
    #     newp = []
    #     for p in range(500, 5001, 500):
    #         newt.append(thr_from_p_sigma(5, p, 500, n * 0.1, 0.95))
    #         newp.append(rho_from_p_sigma(5, p, 500, n * 0.1, 0.02))
    #     thrs.append(newt)
    #     probs.append(newp)
    #
    # np.savetxt("thrs_aa.txt", thrs)
    # np.savetxt("probs_aa.txt", probs)


    # DPNF
    # sigmas = []
    # for eps in np.linspace(0.2, 3.0, 15):
    #     sigma = sigma_from_eps_delta(1e-5, eps, T=6000, total_size=25000, batch_size=100)
    #     print(eps, sigma)
    #     sigmas.append(sigma)
    # print(sigmas)
