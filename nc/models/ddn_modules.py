import torch
import torch.nn as nn   
import torch.nn.functional as F
import numpy as np
from torch.func import jacrev, hessian
from scipy.linalg import qr
from nc.utils import *
from nc.models.model_structure import *

HARDWARE = 'GPU'

if HARDWARE == 'GPU':
    import nc.optimisers.pymanopt.gpu as pymanopt
    from nc.optimisers.pymanopt.gpu import optimizers as manoptim
    from nc.optimisers.pymanopt.gpu import manifolds
else:
    import nc.optimisers.pymanopt.cpu as pymanopt
    from nc.optimisers.pymanopt.cpu import optimizers as manoptim
    from nc.optimisers.pymanopt.cpu import manifolds


'''
Note: The following code is not fully optimised for speed and memory usage.
It is intended for research purposes and to be used as a reference implementation.
We are planning to release a more optimised version in the future.
'''

class ClosestETFGeometryFcn(torch.autograd.Function):
    
    @staticmethod
    def _riemannian_optimisation(P, Y, M, Prox, logfile, feasibility=False):

        d, K = P.shape
        if HARDWARE == 'CPU':
            Y = Y.cpu().double()
            M = M.cpu().double()
            Prox = Prox.cpu().double()
            manifold = manifolds.Stiefel(d, K)
        else:
            manifold = manifolds.Stiefel(d, K, device=P.device)
        
        # delta can be set arbitrarily as long as it's within reasonable bounds
        delta = 1e-3

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            '''
            cost(X) = \|Y - X M\|_F^2 + delta/2 * \|X - Prox\|_F^2
            '''
            if feasibility:
                p = torch.trace(Y.T @ Y) - 2 / pow(K - 1, 0.5) * torch.trace(Y.T @ X @ M) + 1 / (K - 1) * torch.trace(X.T @ X @ M)
            else:
                p = torch.trace(Y.T @ Y) - (2 / pow(K - 1, 0.5)) * torch.trace(Y.T @ X @ M) + 1 / (K - 1) * torch.trace(X.T @ X @ M) \
                        + delta * (0.5 * torch.trace(X.T @ X) - torch.trace(Prox.T @ X) + 0.5 * torch.trace(Prox.T @ Prox))
            
            return p
           

        @pymanopt.function.pytorch(manifold)
        def euclidean_gradient(X):
            if feasibility:
                return 2 / (K - 1) * X @ M - (2 / pow(K - 1, 0.5)) * Y @ M
            else:
                return 2 / (K - 1) * X @ M - (2 / pow(K - 1, 0.5)) * Y @ M + delta * (X - Prox)

        
        @pymanopt.function.pytorch(manifold)
        def euclidean_hessian(X, V):
            if feasibility:
                return 2 / (K - 1) * V @ M
            else:
                return 2 / (K - 1) * V @ M + delta * V


        problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, euclidean_hessian=euclidean_hessian)

        # from pymanopt.tools.diagnostics import check_gradient, check_hessian
        # check_gradient(problem)
        # check_hessian(problem)
        
        optimizer = manoptim.TrustRegions(max_iterations=30, verbosity=0)

        result = optimizer.run(problem, initial_point=P)

        if logfile is not None:
            ClosestETFGeometryFcn._riemannian_opt_stdout(result, M, Prox, logfile)

        if HARDWARE == 'CPU':
            return torch.DoubleTensor(result.point).float()  # if you running gradcheck make sure to return a double tensor
        else: 
            return result.point

    @staticmethod
    def _riemannian_opt_stdout(result, M, Prox, logfile):
        
        print_and_save(f"Check objective minimisation:\n{result.cost}", logfile, save_only=True)
        print_separation_line(logfile, save_only=True)
        print_and_save(f"Stopping Criterion: {result.stopping_criterion}", logfile, save_only=True)
        print_separation_line(logfile, save_only=True)
        print_and_save(f"Completion Time: {result.time}", logfile, save_only=True)
        print_separation_line(logfile, save_only=True)
        print_and_save(
            f"Check optimality measure (L_inf norm):\n{result.gradient_norm}", logfile, save_only=True)
        print_separation_line(logfile, save_only=True)
        print_and_save(f"Difference between ETFs in Frobenius norm:\n{torch.linalg.norm(torch.DoubleTensor(result.point) @ M - Prox@M)}", logfile, save_only=True)

    @staticmethod
    def _solve_blockwise_diagonal_system(A, B, block_size, L=None):
        # Assuming A is a block diagonal matrix and B is a matrix
        # Assuming the blocks in A are of size block_size x block_size

        # Perform Cholesky decomposition on the first block (since all the blocks are the same)
        # Need to make sure that A is symmetric positive definite
        # the code won't raise an error otherwise but it will silently fail
        if L is None:
            L, _ = torch.linalg.cholesky_ex(A)
            # L, pivots, _ = torch.linalg.ldl_factor_ex(A)
        
        B = B.view(-1, block_size, B.shape[-1])

        L_expanded = L.unsqueeze(0).expand(B.shape[0], -1, -1)
        # pivots_expanded = pivots.unsqueeze(0).expand(B.shape[0], -1)

        # Solve the linear system for all blocks at once
        X = torch.cholesky_solve(B, L_expanded)
        # X = torch.linalg.ldl_solve(L_expanded, pivots_expanded, B)
        
        # Reshape solutions to match the original shape
        X = X.reshape(-1, B.shape[-1])

        return X, L

    @staticmethod
    def _solve_general_linear_system(A, B):
        """
        Solve the general linear system AX = B.
        """
        # try cholesky decomposition and if it fails revert to LU solver
        try:
            L, _ = torch.linalg.cholesky_ex(A)
            X = torch.cholesky_solve(B, L)
        except:
            print("Cholesky decomposition failed. Reverting to LU solver.")
            X = torch.linalg.solve(A, B)

        return X

    @staticmethod
    def forward(ctx, feature_means, M, starting_point, Prox, LCI, logfile):
        # compute the closest ETF
        with torch.no_grad():
            
            # This if statement runs only for the first forward pass of the first epoch. This is to establish a good initial point
            if starting_point is None:
                starting_point = torch.eye(feature_means.shape[0], M.shape[0], dtype=torch.float32, device=feature_means.device)

                P = ClosestETFGeometryFcn._riemannian_optimisation(starting_point, feature_means, M, Prox, logfile, feasibility=True)
                starting_point = P.clone().detach()
                Prox = P.clone().detach()
                

            P = ClosestETFGeometryFcn._riemannian_optimisation(starting_point, feature_means, M, Prox, logfile)
        if HARDWARE == 'CPU':
            P = P.to(device=feature_means.device)     

        ctx.save_for_backward(feature_means, M, P.contiguous(), Prox, LCI)
        
        return P

    @staticmethod
    def backward(ctx, grad_output):

        if grad_output is None:
            return None

        # Unpack the saved tensors
        delta = 1e-3
        H, M, P, Prox, LCI = ctx.saved_tensors

        if LCI is None:
            raise ValueError("LCI is None. Please provide a valid LCI matrix. Did you mean to use only the forward pass?")

        d, K = H.shape
        
        v = grad_output.reshape(-1, 1)

        fY = 2 / (K - 1) * P @ M - (2 / pow(K - 1, 0.5)) * H @ M + delta * (P - Prox)
        Id = torch.eye(d, dtype=torch.float32, device=H.device)
                
        Sigma =  0.5 * (fY.T @ P + P.T @ fY)
        

        # # Check proposition for equality constraints problems in DDN paper
        # # Dy(x) = H^{-1}A^T (AH^{-1}A^T)^{-1} AH^{-1}B - H^{-1}B
        # # DJ(x) = DJ(y)Dy(x), and we denote v^T = DJ(y),
        # # so DJ(x) = v^T H^{-1}A^T (AH^{-1}A^T)^{-1} AH^{-1}B - v^T H^{-1}B

        # # We denote w^T = -v^T H^{-1} and t^T =   A H^{-1}
        # # Solve w = -H^{-1} v and t = H^{-1} A^T
        # # We denote and solve s = (A H^-1 A^T)^{-1} A H^{-1} v = -(A t)^{-1} A w
        # # Compute w + ts = -H^-1 v + H^-1 A^T (A H^-1 A^T)^-1 A H^-1 v:
        # # compute (w^T + ts^T) B where B=D^2_{xiy} wrt input x


        ## Uncomment to check for critical point solution 
        # print("Condition 1", torch.linalg.norm(P.T @ fY - fY.T @ P, ord=torch.inf))
        # print("Condition 1 is satisfied: ", torch.allclose(P.T @ fY, fY.T @ P, rtol=0.0, atol=1e-6))
        # print("Condition 2", torch.linalg.norm(fY - P @ P.T @ fY, ord=torch.inf))
        # print("Condition 2 is satisfied: ", torch.allclose(fY, P @ P.T @ fY, rtol=0.0, atol=1e-6))
        # print("Condition 3", torch.linalg.norm(P.T @ P - torch.eye(K, dtype=torch.float32), ord=torch.inf))
        # print("Condition 3 is satisfied: ", torch.allclose(P.T @ P, torch.eye(K, dtype=torch.float32), rtol=0.0, atol=1e-7))

        dec_Hkron_block =  2 / (K - 1) * M + delta * Id[:K, :K] - Sigma
        
        A_kron =  LCI @ torch.einsum('ij,kl->ikjl', P, Id[:K, :K]).view(-1, K * K).T

       
        w, L = ClosestETFGeometryFcn._solve_blockwise_diagonal_system(dec_Hkron_block, -1.0 * v, block_size=K)

        t, _ = ClosestETFGeometryFcn._solve_blockwise_diagonal_system(dec_Hkron_block, A_kron.T, block_size=K, L=L)

        s = ClosestETFGeometryFcn._solve_general_linear_system(A_kron @ t, -1.0 * A_kron @ w)

        uts = w + t @ s

        # Reshape uts to match the shape of Id and M
        # Perform separate matrix multiplications
        H_grad = - 2 / pow(K - 1, 0.5) * uts.view(d, K) @ M

        return H_grad, None, None, None, None, None


class FeaturesMovingAverageLayer(nn.Module):

    def __init__(self, d, K, device='cpu'):
        super().__init__()
        self.d = d
        self.K = K
        self.device = device
        self.FMA_prev = torch.zeros(d, K, requires_grad=False, device=device)
        self.count = 0
        self.alpha = 1
        # Initialise has_sampled_point as a list of False values
        self.has_sampled_point = torch.tensor([False]*K, device=device)
        

    def reset(self):
        self.FMA_prev = torch.zeros(self.d, self.K, requires_grad=False, device=self.device)
        self.count = 0
        self.alpha = 1
        # Reset has_sampled_point to a list of False values
        self.has_sampled_point = torch.tensor([False]*self.K, device=self.device)

    def update(self, val, n=1, method='CMA'):
        if method == 'CMA':
            self.FMA_curr = (self.FMA_prev.to(device=val.device) * self.count + val) / (self.count + n)
        elif method == 'EMA':
            if self.alpha > 1e-4:
                self.alpha = 2 / (self.count + n + 1)
            else:
                self.alpha = 1e-4
            self.FMA_curr = self.alpha * val + (1 - self.alpha) * self.FMA_prev.to(device=val.device)
            # self.alpha = 0.1
        self.FMA_prev = self.FMA_curr.detach()
        self.count += n

    def forward(self, features, targets, method='EMA'):

        mu_G = torch.mean(features, dim=0)

        one_hot_targets = F.one_hot(targets, num_classes=self.K).float()
        # Create a mask for the classes that are present in the batch
        class_mask = one_hot_targets.sum(dim=0) > 0

        # Pre-compute reused values
        one_hot_targets_masked = one_hot_targets[:, class_mask]
        one_hot_targets_masked_sum = one_hot_targets_masked.sum(dim=0)[:, None]

        feature_mean = torch.einsum('ij,ik->jk', one_hot_targets_masked, features) / one_hot_targets_masked_sum
        feature_mean = feature_mean - mu_G

        unique_classes = torch.unique(targets)
        if unique_classes.shape[0] < self.K:
            
            # Create a placeholder for feature_mean with the correct size
            feature_mean_full = torch.full((self.K, feature_mean.shape[1]), float('nan'), device=self.device)

            # Update feature_mean_full with the computed feature_mean values
            feature_mean_full[class_mask, :] = feature_mean

            nan_indices = torch.isnan(feature_mean_full).any(dim=1)
            feature_mean_full[nan_indices, :] = mu_G    

            # Update has_sampled_point for non-NaN values
            self.has_sampled_point = torch.logical_or(self.has_sampled_point, ~nan_indices)

            # If FMA_prev is empty, fill NaN values with mu_G
            if self.FMA_prev.sum() == 0:
                feature_mean_full[nan_indices, :] = mu_G
            else:
                # For NaN values that haven't been sampled yet, fill with mu_G
                unsampled_nan_indices = torch.logical_and(nan_indices, ~self.has_sampled_point)
                feature_mean_full[unsampled_nan_indices, :] = mu_G
                self.FMA_prev.T[unsampled_nan_indices, :] = mu_G

                # For NaN values that have been sampled, fill with previous value
                sampled_nan_indices = torch.logical_and(nan_indices, self.has_sampled_point)
                feature_mean_full[sampled_nan_indices, :] = self.FMA_prev.T[sampled_nan_indices, :]

            feature_mean = feature_mean_full
        
        self.update(feature_mean.T, method=method)

        self.FMA_curr = self.FMA_curr / torch.linalg.matrix_norm(self.FMA_curr)

        return self.FMA_curr, mu_G

class ClosestETFGeometryLayer(nn.Module):

    def __init__(self, dimensions, num_classes, logfile=None, device='cpu'):
        super().__init__()
        self.dimensions = dimensions
        self.K = num_classes    
        self.logfile = logfile
        self.device = device

        self.M = torch.eye(self.K, dtype=torch.float32, device=self.device) - 1 / self.K * torch.ones((self.K, self.K), dtype=torch.float32, device=self.device)
        
        # self.P_init = random_haar_orthogonal_matrix(self.dimensions, self.K, device=device)
        # self.P_init = torch.eye(self.dimensions, self.K, dtype=torch.float32, device=device)
        self.Prox = None
        self.P_init = None

        # if having memory issues, uncomment the following line
        # self.LKI =None

        # and comment the following block
        p = torch.arange(self.K * self.K).reshape((self.K, self.K)).T.ravel()
        Ikk = torch.eye(self.K * self.K, dtype=torch.float32, device=device)
    
        elim_rows = self.K * (self.K + 1) // 2
        elim_cols = self.K * self.K
        Lk = torch.zeros((elim_rows, elim_cols), dtype=torch.float32, device=device)
        idx = torch.tril_indices(self.K, self.K)
        Lk[torch.arange(idx[0].shape[0], device=device), idx[0]*self.K + idx[1]] = 1

        self.LKI = Lk @ (Ikk[p, :] + Ikk)


    def forward(self, feature_means):

        P = ClosestETFGeometryFcn.apply(feature_means, self.M, self.P_init, self.Prox, self.LKI, self.logfile)
        self.Prox = P.clone().detach()
        if HARDWARE == 'CPU':
            self.P_init = P.clone().cpu().double().detach().numpy()
        else:
            self.P_init = P.clone().detach()

        return P
    

#
# --- Test Gradient ---
#


def random_haar_orthogonal_matrix(m, n):
    # if HARDWARE == 'CPU':
    H = np.random.randn(m, m)
    Q, R = qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    P = Q[:, :n]

    # assert np.allclose(P.conj().T @ P, np.eye(n, n)),\
    #     "Permutation matrix P doesn't have orthonormal columns.\n{}".format(
    #         P.conj().T @ P)
            
    return P


def main():
    from torch.autograd import gradcheck
    set_seed(666)
    B, d, K = 5, 10, 5
    features = torch.randn((B, d), dtype=torch.float64)

    targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    # compute the per class mean and global mean of the features
    mu_G = torch.mean(features, dim=0)

    targets = F.one_hot(targets, num_classes=K).double()
    feature_mean = torch.einsum('ij,ik->jk', targets, features) / targets.sum(dim=0)[:, None]
    feature_mean -= mu_G
    H = feature_mean.T

    H = H / torch.linalg.matrix_norm(H)    
    H.requires_grad = True

    ETFGeometry = ClosestETFGeometryFcn.apply
    M = torch.eye(K, dtype=torch.float64) - 1 / K * torch.ones((K, K), dtype=torch.float64)

    Prox = torch.eye(d, K, dtype=torch.float64)

    starting_point = random_haar_orthogonal_matrix(H.shape[0], K)
    
    p = torch.arange(K * K).reshape((K, K)).T.ravel()
    Ikk = torch.eye(K * K, dtype=torch.float64)
    Ik = torch.eye(K, dtype=torch.float64)
    elim_rows = K * (K + 1) // 2
    elim_cols = K * K
    Lk = torch.zeros((elim_rows, elim_cols), dtype=torch.float64)
    idx = torch.tril_indices(K, K)
    Lk[torch.arange(idx[0].shape[0]), idx[0]*K + idx[1]] = 1

    LKI = Lk @ (Ikk[p, :] + Ikk)


    
    P = ETFGeometry(H, M, starting_point, Prox, LKI, None)
    print("Permutation P:\n", P)
    print("Solution:\n", P@M)
    print("Is permutation P orthogonal: ", torch.allclose(P.T @ P, torch.eye(K, dtype=torch.float64), rtol=0.0, atol=1e-6))

    torch.autograd.set_detect_anomaly(True)
    test = gradcheck(ETFGeometry, (H, M, starting_point, Prox, LKI, None), raise_exception=True)
    print("Backward pass: {}".format(test))


if __name__ == '__main__':
    main()