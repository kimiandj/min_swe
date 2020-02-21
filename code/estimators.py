import numpy as np
from utils import swdistance, expected_swdistance


def compute_meswe(X, n_montecarlo, n_generated_samples, n_projections, batch_size,
                  distribution, order=2, n_iterations=10000):
    n_samples, dim = X.shape
    if distribution == 'gaussian':
        # Parameter initialization
        sw_mu = np.ones(dim)
        sw_sigma = 0.01
        # ADAM optimizer: initialization
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        alpha = 1e-3
        M = np.zeros(dim+1)
        R = np.zeros(dim+1)
        j = 0
        for it in range(n_iterations):
            indices = np.random.permutation(n_samples)
            X_mix = X[indices, :]
            for i in range(0, n_samples, batch_size):
                j += 1
                X_i = X_mix[i:i + batch_size]
                # Compute expected SW and its gradient with respect to the parameters
                sw_dist, grad_mu, grad_sigma = expected_swdistance(X_i, distribution, sw_mu, sw_sigma,
                                                                   n_generated_samples, n_montecarlo,
                                                                   L=n_projections, p=order)
                sw_params = np.append(sw_mu, sw_sigma)
                grad = np.append(grad_mu, grad_sigma)
                # Gradient descent
                M = beta1 * M + (1. - beta1) * grad
                R = beta2 * R + (1. - beta2) * grad ** 2
                m_hat = M / (1. - beta1 ** j)
                r_hat = R / (1. - beta2 ** j)
                sw_params = sw_params - alpha * m_hat / (np.sqrt(r_hat) + eps)
                sw_mu = sw_params[:dim]
                sw_sigma = sw_params[-1]
        return sw_mu, sw_sigma
    # MESWE for elliptically contoured stable parameter using ADAM algorithm
    elif distribution == 'ellipstable':        
        # Parameter initialization
        sw_mu = np.zeros(dim)
        # ADAM optimizer: initialization
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        alpha = 1e-3
        M = np.zeros(dim)
        R = np.zeros(dim)
        j = 0
        for it in range(n_iterations):
            indices = np.random.permutation(n_samples)
            X_mix = X[indices, :]
            for i in range(0, n_samples, batch_size):
                j += 1
                X_i = X_mix[i:i + batch_size]
                # Compute expected SW and its gradient with respect to the parameters
                sw_dist, grad_mu = expected_swdistance(X_i, distribution, sw_mu, 1., 
                                                       n_generated_samples, n_montecarlo, 
                                                       L=n_projections, p=order)
                # Gradient descent
                M = beta1 * M + (1. - beta1) * grad_mu
                R = beta2 * R + (1. - beta2) * grad_mu ** 2
                m_hat = M / (1. - beta1 ** j)
                r_hat = R / (1. - beta2 ** j)
                sw_mu = sw_mu - alpha * m_hat / (np.sqrt(r_hat) + eps)
        return sw_mu
    else:
        raise Exception("Error: 'distribution' parameter should be 'gaussian' or 'ellipstable'.")


def compute_gaussian_mswe(X, n_projections, batch_size, order=2, n_iterations=20000):
    n_samples, dim = X.shape
    # Parameter initialization
    sw_mu = np.ones(dim)
    sw_sigma = 0.01
    # ADAM optimizer: initialization
    batch_size = batch_size
    n_iterations = n_iterations
    beta1 = .9
    beta2 = .999
    eps = 1e-8
    alpha = 1e-3
    M = np.zeros(dim+1)
    R = np.zeros(dim+1)
    j = 0
    for it in range(n_iterations):
        indices = np.random.permutation(n_samples)
        X_mix = X[indices, :]
        for i in range(0, n_samples, batch_size):
            j += 1
            X_i = X_mix[i:i + batch_size]
            # Compute SW and its gradient with respect to the parameters
            sw_dist, grad_mu, grad_sigma = swdistance(X_i, sw_mu, sw_sigma * np.eye(dim),
                                                      L=n_projections, p=order)
            sw_params = np.append(sw_mu, sw_sigma)
            grad = np.append(grad_mu, grad_sigma)
            # Gradient descent
            M = beta1 * M + (1. - beta1) * grad
            R = beta2 * R + (1. - beta2) * grad ** 2
            m_hat = M / (1. - beta1 ** j)
            r_hat = R / (1. - beta2 ** j)
            sw_params = sw_params - alpha * m_hat / (np.sqrt(r_hat) + eps)
            sw_mu = sw_params[:dim]
            sw_sigma = sw_params[-1]
    return sw_mu, sw_sigma
