import autograd.numpy as np
np.random.seed(0)


class GSWVect:
    def __init__(self, n_projections=10, order=2):
        self.n_projections = n_projections
        self.order = order

    def compute_gsw(self, X, sw_mu, sw_sigma, theta=None, proj=True, requires_grad=False):
        """
        Computes Sliced-Wasserstein distance of order 2 between two empirical distributions.
        Note that the number of samples is assumed to be equal (This is however not necessary
        and could be easily extended for empirical distributions with different number of samples)
        :param X:  stacks of samples from the first distribution (M x N x d matrix, with M the number of stacks,
                    N the number of samples and d the dimension)
        :param Y:  stacks of samples from the second distribution (M x N x d matrix, with M the number of stacks,
                    N the number of samples and d the dimension)
        :param theta:  stacks of directions of projections (M x L x d matrix, with M the number of stacks, L the number
                    of directions and d the dimension)
        :return:  the sliced-Wasserstein distance between X[i] and Y[i] for i in 1,...,M (vector of size M)
        """
        M, N, d = X.shape
        n_montecarlo = M
        n_generated_samples = N
        gamma = sw_sigma * np.eye(d)
        U = np.random.normal(size=(n_montecarlo, n_generated_samples, d))
        Y = sw_mu + np.einsum('nij,njk->nik', U, gamma.T[np.newaxis, :])

        M_Y, N_Y, d_Y = Y.shape
        assert d == d_Y and M == M_Y

        order = self.order

        if proj:
            if theta is None:
                theta = self.random_slice(M, d)
            Xslices = self.get_slice(X, theta)
            Yslices = self.get_slice(Y, theta)
        else:
            Xslices, Yslices = X, Y

        Xslices_sorted = np.sort(Xslices, axis=1)
        indices_sorted = np.argsort(Yslices, axis=1)
        Yslices_sorted = np.take_along_axis(Yslices, indices_sorted, axis=1)

        if N == N_Y:
            diff = Xslices_sorted - Yslices_sorted
            sw_dist = np.sum(np.abs(diff) ** order, (1, 2)) / (self.n_projections*N)
            sw_dist = sw_dist.mean()
            if requires_grad:
                theta_U = self.get_slice(U, theta)
                theta_U = np.take_along_axis(theta_U, indices_sorted, axis=1)
                replicate_theta = (np.stack([theta] * Xslices.shape[1])).transpose((1, 0, 2, 3))
                sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * replicate_theta
                sw_grad_sigma = - order * diff ** (order - 1) * theta_U
                if order % 2 == 1:
                    sw_grad_mu = - order * (diff[:, :, :, np.newaxis]) ** (order - 1) * (
                                diff[:, :, :, np.newaxis] / np.abs(diff[:, :, :, np.newaxis])) * replicate_theta
                    sw_grad_sigma = - order * diff ** (order - 1) * (diff / np.abs(diff)) * theta_U
                sw_grad_mu = (sw_grad_mu.reshape(-1, sw_grad_mu.shape[-1])).mean(axis=0)
                sw_grad_sigma = sw_grad_sigma.mean()
                return sw_dist, sw_grad_mu, sw_grad_sigma
            else:
                return sw_dist
        else:
            n_quantiles = 100
            discretization_quantiles = np.linspace(0, 1, n_quantiles + 2)
            discretization_quantiles = discretization_quantiles[1:-1]

            # With linear interpolation
            positions = (N-1)*discretization_quantiles
            floored = np.floor(positions).astype(int)
            ceiled = floored + 1
            ceiled[ceiled > N-1] = N-1
            weight_ceiled = positions - floored
            weight_floored = 1.0 - weight_ceiled

            d0 = Xslices_sorted[:, :, floored] * weight_floored[np.newaxis, np.newaxis, :]
            d1 = Xslices_sorted[:, :, ceiled] * weight_ceiled[np.newaxis, np.newaxis, :]
            X_empirical_qf = d0 + d1

            positions = (N_Y-1)*discretization_quantiles
            floored = np.floor(positions).astype(int)
            ceiled = floored + 1
            ceiled[ceiled > N_Y-1] = N_Y-1
            weight_ceiled = positions - floored
            weight_floored = 1.0 - weight_ceiled

            d0 = Yslices_sorted[:, :, floored] * weight_floored[np.newaxis, np.newaxis, :]
            d1 = Yslices_sorted[:, :, ceiled] * weight_ceiled[np.newaxis, np.newaxis, :]
            Y_empirical_qf = d0 + d1

            return (np.sum(np.abs(X_empirical_qf - Y_empirical_qf) ** self.order, (1, 2)) / (self.n_projections * n_quantiles)) ** (1/self.order)

    def get_slice(self, X, theta):
        """
        Slices the samples X produced from distribution P_X
        :param X:  stacks of data samples (M x N x d matrix with M the number of stacks, N the number of samples
                    and d the dimension)
        :param theta:  directions of projections (M x L x d matrix with M the number of stacks, L the number of
                    projections and d the dimension)
        :return:  projections of each sample of X along each direction of theta (M x N x L matrix)
        """
        if len(theta.shape) == 1:
            return np.matmul(X, theta)
        else:
            return np.matmul(X, np.transpose(theta, (0, 2, 1)))

    def random_slice(self, M, d):
        """
        Randomly picks different stacks of directions of projections
        :param M:  number of stacks
        :param d:  dimension for each direction
        :return:  stacks of directions of projections (M x L x d matrix with L = n_projections the number of directions)
        """
        theta = np.random.randn(M, self.n_projections, d)
        theta = theta / (np.sqrt((theta ** 2).sum(axis=2)))[:, :, None]
        return theta
