import os
num_numpy_threads = '16'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

import pathlib

import numpy as np
import torch

from nc.utils import *
from nc.args import parse_train_args

import pickle
import matplotlib.pyplot as plt
from scipy.linalg import qr, eigvals
from seaborn import histplot
import nc.optimisers.pymanopt.cpu as pymanopt
from nc.optimisers.pymanopt.cpu import optimizers as manoptim
from nc.optimisers.pymanopt.cpu import manifolds



def riemannian_optimisation(P, Y, logfile, **kwargs):


    d, K = P.shape
    
    M = torch.eye(K, dtype=torch.float64) - 1 / K * torch.ones((K, K), dtype=torch.float64)

    manifold = manifolds.Stiefel(d, K)
    Y = torch.tensor(Y, dtype=torch.float64)

    @pymanopt.function.pytorch(manifold)
    def cost(X):
        return torch.trace(Y.T @ Y) - (2 / pow(K - 1, 0.5)) * torch.trace(Y.T @ X @ M) + 1 / (K - 1) * torch.trace(X.T @ X @ M)

    @pymanopt.function.pytorch(manifold)
    def euclidean_gradient(X):
        return 2 / (K - 1) * X @ M - (2 / pow(K - 1, 0.5)) * Y @ M
    
    @pymanopt.function.pytorch(manifold)
    def euclidean_hessian(X, V):
        return 2 / (K - 1) * V @ M

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, euclidean_hessian=euclidean_hessian)

    optimiser = manoptim.TrustRegions(max_iterations=500, min_step_size=1e-6, verbosity=0)
    result = optimiser.run(problem, initial_point=P)
    
    ETF_path = kwargs['path'] + '/ETFs/'
    pathlib.Path(ETF_path).mkdir(parents=True, exist_ok=True)
    np.save(ETF_path + 'closest_ETF_to_' + kwargs['reference'] + '_' +
            str(kwargs['epoch'] + 1).zfill(3) + '.npy', result.point)

    riemannian_opt_stdout(P, result, logfile)
    
    return result.point

def riemannian_opt_stdout(initial, result, logfile):
    print_and_save(f"Random/Initial starting permutation P:\n{initial}", logfile)
    print_separation_line(logfile)
    d, K = initial.shape
    P_opt = result.point

    print_and_save(f"Optimised permutation P_opt:\n{P_opt}", logfile)
    print_separation_line(logfile)
    print_and_save(f"Check objective minimisation:\n{result.cost}", logfile)
    print_separation_line(logfile)
    print_and_save(f"Stopping Criterion: {result.stopping_criterion}", logfile)
    print_separation_line(logfile)
    print_and_save(f"Completion Time: {result.time}", logfile)
    print_separation_line(logfile)
    print_and_save(
        f"Check feasibility of the constraint (P_opt.T @ P_opt = I):\n{P_opt.T @ P_opt}", logfile)
    print_separation_line(logfile)
    print_and_save(
        f"Check optimality measure (L_inf norm):\n{result.gradient_norm}", logfile)
    print_separation_line(logfile)


def canonical_orthogonal_matrix(m, n):
    return torch.eye(m, n)

def random_haar_orthogonal_matrix(m, n, check_haar=False):
    H = np.random.randn(m, m)
    Q, R = qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    
    # H = torch.randn(m, m, dtype=torch.float64)
    # Q, R = torch.linalg.qr(H)
    # Q = Q @ torch.diag(torch.sign(torch.diag(R)))

    P = Q[:, :n]

    assert np.allclose(P.conj().T @ P, np.eye(n, n)),\
        "Permutation matrix P doesn't have orthonormal columns.\n{}".format(
            P.conj().T @ P)

    # assert torch.allclose(P.T @ P, torch.eye(n, dtype=torch.float64)),\
    #     "Permutation matrix P doesn't have orthonormal columns.\n{}".format(
    #         P.conj().T @ P)

    if check_haar:
        test_haar_measure(m)

    return P


def test_haar_measure(n):
    '''
    TODO: Need to check an issue in the eigenvalues of Haar measure.
    '''
    repeats = 10000

    angles = []
    angles_modified = []
    for rp in range(repeats):
        H = np.random.randn(n, n)
        # H = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
        Q, R = qr(H)
        angles.append(np.angle(eigvals(Q)))
        Q_modified = Q @ np.diag(np.sign(np.diag(R)))

        angles_modified.append(np.angle(eigvals(Q_modified)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    histplot(np.asarray(angles).flatten(), kde=False,
             line_kws=dict(edgecolor="k", linewidth=2), ax=ax[0])
    ax[0].set(xlabel='phase', title='direct')
    histplot(np.asarray(angles_modified).flatten(), kde=False,
             line_kws=dict(edgecolor="k", linewidth=2), ax=ax[1])
    ax[1].set(xlabel='phase', title='modified')
    plt.show()


def find_closest_trained_ETFs(args):
    '''
    '''
    epochs = args.epochs
    logpath = args.save_path + '/logs/'
    pathlib.Path(logpath).mkdir(parents=True, exist_ok=True)

    path = logpath + 'closest_ETF_to_W_and_H_optimisation.txt'
    if os.path.exists(path) and args.skip:
        print('We have already found the closest trained ETFs. Skipping this operation...')
        return

    PATH_TO_INFO_NC_VARIABLES = args.save_path + "/metrics/"

    with open(PATH_TO_INFO_NC_VARIABLES + args.nc_metrics + "_NC_variables.pkl", 'rb') as f:
        info_nc_variables = pickle.load(f)

    Ys_W = info_nc_variables['W']
    Ys_H = info_nc_variables['H']
   
    d, K = Ys_H[0].shape

    P = random_haar_orthogonal_matrix(d, K)

    logfile = open(path, 'w')
    index = 0
    for epoch in range(0, epochs, args.log_interval):
        # Normalise the features and weights
        Y_T_W = Ys_W[index].T
        Y_T_W /= np.linalg.norm(Y_T_W, ord='fro')

        Y_T_H = Ys_H[index]
        Y_T_H /= np.linalg.norm(Y_T_H, ord='fro')
        # Optimisation
        print_separation_line(logfile, 2)
        print_and_save(f"Measure ETF closeness for epoch {epoch + 1}", logfile)
        print_separation_line(logfile, 2)
        
        print_and_save(f"Optimising permutation P with respect to the classifier weights W", logfile)
        P = riemannian_optimisation(P, Y_T_W, logfile, epoch=epoch, path=args.save_path, reference='W')
        print_and_save(f"Optimising permutation P with respect to the penultimate layer features H", logfile)
        P = riemannian_optimisation(P, Y_T_H, logfile, epoch=epoch, path=args.save_path, reference='H')
        index += 1        

    logfile.close()

def distance_between_weight_feature_ETFs(args):
    
    figure_path = args.save_path + '/figures/'
    pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)

    path = args.save_path + f'/logs/W-ETFs_vs_H-ETFs_distance.txt'
    if os.path.exists(path) and args.skip:
        print('We have already measured the distance between ETF closest to weights W and feature-means H^. Skipping this operation...')
        return

    ETF_WH_dists = []
    WH_dists = []
    WH_MM_dists = []

    PATH_TO_INFO_NC_VARIABLES = args.save_path + "/metrics/" + args.nc_metrics + "_NC_variables.pkl"
    with open(PATH_TO_INFO_NC_VARIABLES, 'rb') as f:
        trained_info_nc_variables = pickle.load(f)

    logfile = open(path, 'w')
    index = 0
    for epoch in range(0, args.epochs, args.log_interval):
        if not os.path.exists (args.save_path + \
                                  f"/ETFs/closest_ETF_to_W_{str(epoch + 1).zfill(3)}.npy") or not os.path.exists(args.save_path + f"/ETFs/closest_ETF_to_H_{str(epoch + 1).zfill(3)}.npy"):
            return
        
        P_opt_W_trained = np.load(args.save_path +
                                  f"/ETFs/closest_ETF_to_W_{str(epoch + 1).zfill(3)}.npy")
        P_opt_H_trained = np.load(args.save_path +
                                  f"/ETFs/closest_ETF_to_H_{str(epoch + 1).zfill(3)}.npy")
        d, K = P_opt_W_trained.shape

        M = (np.eye(K) - 1 / K * np.ones((K, K))) / pow(K - 1, 0.5)


        ETF_W_trained = P_opt_W_trained @ M
        ETF_H_trained = P_opt_H_trained @ M

        ETF_WH_dist = np.linalg.norm(ETF_W_trained - ETF_H_trained, ord='fro')**2
        ETF_WH_dists.append(ETF_WH_dist)

        W_trained = trained_info_nc_variables['W'][index].T
        W_trained /= np.linalg.norm(W_trained, ord='fro')
        H_trained = trained_info_nc_variables['H'][index]
        H_trained /= np.linalg.norm(H_trained, ord='fro')

        WH_dist = np.linalg.norm(W_trained - H_trained, ord='fro')**2
        WH_dists.append(WH_dist)

        WH = trained_info_nc_variables['W'][index] @ trained_info_nc_variables['H'][index]
        WH /= np.linalg.norm(WH, ord='fro')
        WH_MM_dist = np.linalg.norm(WH - M, ord='fro')
        WH_MM_dists.append(WH_MM_dist)
        index += 1

        print_separation_line(logfile, 2)
        print_and_save(f"ETF for epoch {epoch+1}/{args.epochs}", logfile)
        print_separation_line(logfile, 2)

        print_and_save(
            f"Distance between the closest optimal W-ETF and closest optimal H-ETF ({epoch + 1}): {ETF_WH_dist}", logfile)
        print_and_save(
            f"Distance between the optimal W and optimal H ({epoch + 1}): {WH_dist}", logfile)
        print_and_save(
            f"Distance between the optimal WH and closest ETF ({epoch + 1}): {WH_MM_dist}", logfile)
        

def convert_value(args, value, ticks=5):
        num_elements = args.epochs // args.log_interval 
        new_step = args.epochs // ticks
        cur_step = num_elements // ticks
        return int((value + cur_step) * new_step / cur_step - new_step)


def build_settings():
    args = parse_train_args()
     
    set_seed(manualSeed=args.seed)
    device = torch.device("cuda:" + str(args.gpu_id)
                          if torch.cuda.is_available() else "cpu")
    args.device = device

    return args


def main():
    args = build_settings()

    find_closest_trained_ETFs(args)
    distance_between_weight_feature_ETFs(args)

if __name__ == "__main__":
    main()
    