# Some parts of this code are taken from https://github.com/skolouri/swgmm

import numpy as np
from scipy import interp


def pWasserstein(I0, I1, p):
    """Given two one-dimensional pdfs I_0 and I_1, this function calculates the following:

    f:   Transport map between I0 and I1, such that f'I_1(f)=I_0
    phi: The transport displacement potential f(x)=x-\nabla phi(x)
    Wp:  The p-Wasserstein distance
    """
    assert I0.shape == I1.shape
    eps = 1e-7
    I0 = I0 + eps  # Add a small value to pdfs to ensure positivity everywhere
    I1 = I1 + eps
    I0 = I0 / I0.sum()  # Normalize the inputs to ensure that they are pdfs
    I1 = I1 / I1.sum()
    J0 = np.cumsum(I0)  # Calculate the CDFs
    J1 = np.cumsum(I1)
    # Here we calculate transport map f(x)=x-u(x)
    x = np.asarray(range(len(I0)))
    xtilde = np.linspace(0, 1, len(I0))
    XI0 = interp(xtilde, J0, x)
    XI1 = interp(xtilde, J1, x)
    u = interp(x, XI0, XI0 - XI1)  # u(x)
    f = x - u
    phi = np.cumsum(u / (len(I0)))  # Integrate u(x) to obtain phi(x)
    phi -= phi.mean()  # Subtract the mean of phi to account for the unknown constant
    Wp = (((abs(u) ** p) * I0).mean()) ** (1.0 / p)
    return f, phi, Wp


def gaussKernel(t, proj):
    density = np.histogram(proj, bins=len(t), range=(t.min(), t.max()))[0]
    return density / float(density.sum())


def gaussian1d(t, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def swdistance(X, mu_, Sigma_, L=180, p=2):
    N, d = X.shape
    theta = np.random.randn(L, d)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]
    T = 1000
    t = np.linspace(-np.abs(X).max() * np.sqrt(2 * d), np.abs(X).max() * np.sqrt(2 * d), T)
    xproj = np.matmul(X, theta.T)
    projectedSigma = np.zeros(L)
    projectedMu = np.zeros(L)
    for l, th in enumerate(theta):
        projectedSigma[l] = np.sqrt(np.matmul(np.matmul(th, Sigma_), th))
        projectedMu[l] = np.matmul(th, mu_)

    sw = 0
    phi = np.zeros((T, L))
    for l in range(L):
        RIx = gaussian1d(t, projectedMu[l], projectedSigma[l])
        RIy = gaussKernel(t, xproj[:, l])
        _, phi_, w2 = pWasserstein(RIx, RIy, p=p)
        phi[:, l] = phi_
        sw += (w2 ** p) / float(L)

    Ixkl = np.array([gaussian1d(t, projectedMu[l], projectedSigma[l]) for l in range(L)]).T
    tSt = np.tile(projectedSigma ** 2, (T, 1))
    tmm = np.tile(t, (L, 1)).T - np.tile(projectedMu, (T, 1))

    # Calculate gradient with respect to \mu (Mean)
    dSWdMu = (np.tile(((phi * Ixkl * tmm / tSt).mean(0)), (d, 1)).T * theta).mean(0)

    # Calculate gradient with respect to \sigma (Variance)
    dsWdSigmaCoeff = (phi * Ixkl * (1.0 / (2 * tSt ** 2)) * ((tmm ** 2) / tSt - 1)).mean(0)
    dSWdSigma_ = [dsWdSigmaCoeff[i] * np.matmul(np.expand_dims(th, 1).T, np.expand_dims(th, 1))
                  for i, th in enumerate(theta)]
    dSWdSigma = np.asarray(dSWdSigma_).mean(0)

    return sw, dSWdMu, dSWdSigma


def expected_swdistance(X, distribution, mu_, sigma2_, N_Y, n_montecarlo=20, L=180, p=2):
    X = np.stack([X] * n_montecarlo)
    M, N, d = X.shape
    order = p
    
    if distribution == 'gaussian':
        Gamma = np.sqrt(sigma2_) * np.eye(d)  # we assume covariance = sigma^2 * I
        U = np.random.normal(size=(n_montecarlo, N_Y, d))
        Y = np.einsum('nij,njk->nik', U, Gamma.T[np.newaxis, :]) + mu_
        # Project data
        theta = np.random.randn(M, L, d)
        theta = theta / (np.sqrt((theta ** 2).sum(axis=2)))[:, :, None]
        theta = np.transpose(theta, (0, 2, 1))
        xproj = np.matmul(X, theta)
        yproj = np.matmul(Y, theta)
        # Compute quantiles
        T = 100
        t = np.linspace(0, 100, T + 2)
        t = t[1:-1]
        xqf = (np.percentile(xproj, q=t, axis=1))
        yqf = (np.percentile(yproj, q=t, axis=1))
        # Compute expected SW distance and its gradient
        diff = (xqf - yqf).transpose((1, 0, 2))
        uqf = ((np.matmul(mu_, theta) - yqf)/sigma2_).transpose((1, 0, 2))  # for each proj
        stack_theta = (np.stack([theta] * T)).transpose((1, 0, 3, 2))
        if order % 2 == 1:
            sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * (
                diff[:, :, :, np.newaxis] / np.abs(diff[:, :, :, np.newaxis])) * stack_theta
            sw_grad_sigma = order * diff ** (order - 1) * (diff / np.abs(diff)) * uqf
        else:
            sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * stack_theta
            sw_grad_sigma = diff ** (order - 1) * uqf
        sw_dist = (np.abs(diff) ** order).mean()
        sw_grad_mu = (sw_grad_mu.reshape(-1, sw_grad_mu.shape[-1])).mean(0)
        sw_grad_sigma = sw_grad_sigma.mean()
        return sw_dist, sw_grad_mu, sw_grad_sigma
    elif distribution == 'ellipstable':
        # Generate data from elliptically contoured stable distribution
        Gamma = np.eye(d)
        alpha = 1.8
        scale = 2. * np.cos(np.pi * alpha / 4.) ** (2. / alpha)
        A = scale * generate_std_1d_alphastable(alpha=alpha/2., beta=1., size=(M, N_Y, 1))
        U = np.random.normal(size=(M, N_Y, d))
        G = np.einsum('nij,njk->nik', U, Gamma.T[np.newaxis, :])
        Y = np.sqrt(A) * G + mu_
        # Project data
        theta = np.random.randn(M, L, d)
        theta = theta / (np.sqrt((theta ** 2).sum(axis=2)))[:, :, None]
        theta = np.transpose(theta, (0, 2, 1))
        xproj = np.matmul(X, theta)
        yproj = np.matmul(Y, theta)
        # Compute quantiles
        T = 100
        t = np.linspace(0, 100, T + 2)
        t = t[1:-1]
        xqf = (np.percentile(xproj, q=t, axis=1))
        yqf = (np.percentile(yproj, q=t, axis=1))
        # Compute expected SW distance and its gradient
        diff = (xqf - yqf).transpose((1, 0, 2))
        stack_theta = (np.stack([theta] * T)).transpose((1, 0, 3, 2))
        if order % 2 == 1:
            sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * (
                diff[:, :, :, np.newaxis] / np.abs(diff[:, :, :, np.newaxis])) * stack_theta
        else:
            sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * stack_theta
        sw_dist = (np.abs(diff) ** order).mean()
        sw_grad_mu = (sw_grad_mu.reshape(-1, sw_grad_mu.shape[-1])).mean(0)
        return sw_dist, sw_grad_mu        
    else:
        raise Exception("Error: 'distribution' parameter should be 'gaussian' or 'ellipstable'.")


def generate_std_1d_alphastable(alpha, beta, size):
    """ 
    Generates samples from the univariate stable distribution
    S(alpha, beta, scale=1, mu=0)
    """
    zeta = - beta * np.tan(np.pi * alpha / 2.0)
    U = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=size)
    W = np.random.exponential(size=size)
    if alpha == 1.0:
        ksi = np.pi / 2
        Xa = (np.pi / 2 + beta * U) * np.tan(U)
        Xb = beta * np.log((np.pi / 2 * W * np.cos(U)) / (np.pi / 2 + beta * U))
        X = 1 / ksi * (Xa - Xb)
    else:
        ksi = 1 / alpha * np.arctan(-zeta)
        Xa = np.sin(alpha * (U + ksi)) / np.cos(U) ** (1 / alpha)
        Xb = np.cos(U - alpha * (U + ksi)) / W
        power1 = (1 - alpha) / alpha
        power2 = 1 / (2 * alpha)
        X = Xa * (Xb ** power1)
        X = (1 + zeta ** 2) ** power2 * X
    return X
