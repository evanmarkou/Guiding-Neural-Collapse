import numpy as np
import pickle
import pathlib
from nc.args import parse_eval_args
from nc.utils import print_and_save, print_separation_line

NUMBER_OF_SEEDS = 3

def build_settings():
    args = parse_eval_args()

    args.load_path = args.storage + '/experiments/' + args.dataset + '/' + args.model + \
            '/model_weights/' + str(args.uid) + "/seed-"
    
    args.output_path = args.storage + '/experiments/' + args.dataset + '/' + args.model + \
            '/model_weights/' + str(args.uid) + "/seed_statistics/" + args.nc_metrics + "/"

    args.info_nc_metrics_seeds = []
    args.info_nc_variables_seeds = []

    # loop over seeds and pickle load the objects
    for seed in range(NUMBER_OF_SEEDS):
        PATH_TO_INFO_NC_METRICS = args.load_path + str(seed + 1) + '/' + args.nc_metrics + \
        "/metrics/" + args.nc_metrics + "_NC_metrics.pkl"

        PATH_TO_INFO_NC_VARIABLES = args.load_path + str(seed + 1) + '/' + args.nc_metrics + \
        "/metrics/" + args.nc_metrics + "_NC_variables.pkl"

        with open(PATH_TO_INFO_NC_METRICS, 'rb') as f:
            args.info_nc_metrics_seeds.append(pickle.load(f))

        with open(PATH_TO_INFO_NC_VARIABLES, 'rb') as f:
            args.info_nc_variables_seeds.append(pickle.load(f))

    return args

def stack_seed_statistics(seed_statistics):
    """
    Stack seed statistics into a numpy array.

    Args:
        seed_statistics (dict): Dictionary of lists seed statistics.

    Returns:
        stacked_seed_statistics (np.ndarray): stacked seed statistics.
    """
    for key, seed_stat in seed_statistics.items():
        seed_statistics.update({key: np.array(seed_stat)})
    return seed_statistics


def compute_mean_and_std(seed_statistics):
    """
    Compute mean and standard deviation of seed statistics.

    Args:
        seed_statistics (dict): Dictionary of arrays of seed statistics.

    Returns:
        seed_stats (dict): A new dictionary with mean and standard deviation of each key of seed statistics.
    """
    seed_stats = {}
    for key, seed_stat in seed_statistics.items():
        # if seed_stat is empty continue
        if seed_stat.size == 0:
            continue
        seed_stats.update({key: {"mean": np.mean(seed_stat, axis=0), "std": np.std(seed_stat, ddof=1, axis=0), 
                           'median': np.median(seed_stat, axis=0),
                           "max": np.max(seed_stat, axis=0), "min": np.min(seed_stat, axis=0)}})
    return seed_stats

def save_seed_statistics(seed_statistics, output_path):
    """
    Save the new seed statistics which contain the mean and std of each key of seed statistics, using pickle.

    Args:
        seed_statistics (dict): Dictionary of arrays of seed statistics.
        output_path (str): Output path.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(seed_statistics, f, protocol=pickle.HIGHEST_PROTOCOL)

def log_seed_statistics(seed_statistics, output_path):
    """
    Log the new seed statistics which contain the mean and std of each key of seed statistics.

    Args:
        seed_statistics (dict): Dictionary of arrays of seed statistics.
        output_path (str): Output path.
    """
    logfile = open(output_path, 'w')
    print_and_save(f"Seed statistics: {output_path}", logfile)
    print_and_save(f"Number of seeds: {NUMBER_OF_SEEDS}", logfile)
    print_separation_line(logfile, symbol='=')
    for key, seed_stat in seed_statistics.items():
        print_and_save(f"Seed statistics: {key}", logfile)
        print_and_save(f"Mean:\n {seed_stat['mean']}", logfile)
        print_and_save(f"Standard deviation:\n {seed_stat['std']}", logfile)
        print_and_save(f"Median:\n {seed_stat['median']}", logfile)
        print_and_save(f"Max:\n {seed_stat['max']}", logfile)
        print_and_save(f"Min:\n {seed_stat['min']}", logfile)
        print_separation_line(logfile)

    logfile.close()


def aggregate_seed_statistics(args):
    """
    Aggregate seed statistics.
    Break down into four parts
    1. Collect
    2. Stack
    3. Compute mean and std
    4. Save mean and std

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    seed_metrics = {
        "top1": [],
        "top5": [],
        "NCC_mismatch": [],
        "loss": [],
        "cos_sim": [],
        "WH_norms": [],
        "H_norms": [],
        "W_norms": [],
        "numerical_rank": [],
        "avg_cos_margin": [],
        "cos_margin_distr": [],
        "NC1": [],
        "NC2": [],
        "NC3": [],
        "NC4": []
    }

    seed_variables = {
        "W_ETF_dists": [],
        "H_ETF_dists": [],
        "ETF_WH_dists": [],
        "ETF_ref_W_dists": [],
        "ETF_ref_H_dists": []
    }

    # Collect seed statistics
    for seed in range(NUMBER_OF_SEEDS):
        # Collect seed metrics
        seed_metrics["top1"].append(args.info_nc_metrics_seeds[seed]['acc1'])
        seed_metrics["top5"].append(args.info_nc_metrics_seeds[seed]['acc5'])
        seed_metrics['NCC_mismatch'].append(args.info_nc_metrics_seeds[seed]['NCC_mismatch'])
        if args.nc_metrics == 'train':
            seed_metrics['loss'].append(np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + "/metrics/loss.npy"))
            seed_metrics['cos_sim'].append(np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + "/metrics/cos_sim.npy"))
        seed_metrics['numerical_rank'].append(args.info_nc_metrics_seeds[seed]['numerical_rank_metric'])
        seed_metrics['avg_cos_margin'].append(args.info_nc_metrics_seeds[seed]['avg_cos_margin'])
        seed_metrics['cos_margin_distr'].append(args.info_nc_metrics_seeds[seed]['all_cos_margin'])
        seed_metrics['WH_norms'].append(args.info_nc_metrics_seeds[seed]['WH_equinorm_diff_metric'])
        seed_metrics['W_norms'].append(args.info_nc_metrics_seeds[seed]['W_equinorm_metric'])
        seed_metrics['H_norms'].append(args.info_nc_metrics_seeds[seed]['H_equinorm_metric'])
        seed_metrics['NC1'].append(args.info_nc_metrics_seeds[seed]['collapse_metric'])
        seed_metrics['NC2'].append(args.info_nc_metrics_seeds[seed]['ETF_metric'])
        seed_metrics['NC3'].append(args.info_nc_metrics_seeds[seed]['WH_relation_metric'])
        seed_metrics['NC4'].append(args.info_nc_metrics_seeds[seed]['Wh_b_relation_metric'])

        # Collect seed variables
        W_opt = args.info_nc_variables_seeds[seed]['W']
        H_opt = args.info_nc_variables_seeds[seed]['H']
        
        
        ETF_WH_dists = []
        W_ETF_dists = []
        H_ETF_dists = []
        ETF_ref_W_dists = []
        ETF_ref_H_dists = []
        index = 0
        # Get ETF reference
        P_opt_W_ref = np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + f"/ETFs/closest_ETF_to_W_{str(args.ref_ETF).zfill(3)}.npy")
        P_opt_H_ref = np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + f"/ETFs/closest_ETF_to_H_{str(args.ref_ETF).zfill(3)}.npy")
    
        d, K = P_opt_W_ref.shape
        M = (np.eye(K) - 1 / K * np.ones((K, K))) / pow(K - 1, 0.5)

        ETF_reference_W = P_opt_W_ref @ M
        ETF_reference_H = P_opt_H_ref @ M

        # Collect ETF distances
        for epoch in range(0, args.epochs, args.log_interval):
            P_opt_W = np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + f"/ETFs/closest_ETF_to_W_{str(epoch + 1).zfill(3)}.npy")
            P_opt_H = np.load(args.load_path + str(seed + 1) + '/' + args.nc_metrics + f"/ETFs/closest_ETF_to_H_{str(epoch + 1).zfill(3)}.npy")

            ETF_W = P_opt_W @ M
            ETF_H = P_opt_H @ M

            ETF_WH_dist = np.linalg.norm(ETF_W - ETF_H, ord='fro')**2
            ETF_WH_dists.append(ETF_WH_dist)
            
            ETF_W_ref_dist = np.linalg.norm(ETF_W - ETF_reference_W, ord='fro')**2
            ETF_ref_W_dists.append(ETF_W_ref_dist)

            ETF_H_ref_dist = np.linalg.norm(ETF_H - ETF_reference_H, ord='fro')**2
            ETF_ref_H_dists.append(ETF_H_ref_dist)

            W = W_opt[index].T / np.linalg.norm(W_opt[index].T, ord='fro')
            H = H_opt[index] / np.linalg.norm(H_opt[index], ord='fro')

            ETF_W_dist = np.linalg.norm(ETF_W - W, ord='fro')**2
            W_ETF_dists.append(ETF_W_dist)

            ETF_H_dist = np.linalg.norm(ETF_H - H, ord='fro')**2
            H_ETF_dists.append(ETF_H_dist)

            index += 1
        
        seed_variables['ETF_WH_dists'].append(ETF_WH_dists)
        seed_variables['ETF_ref_W_dists'].append(ETF_ref_W_dists)
        seed_variables['ETF_ref_H_dists'].append(ETF_ref_H_dists)
        seed_variables['W_ETF_dists'].append(W_ETF_dists)
        seed_variables['H_ETF_dists'].append(H_ETF_dists)
           

    # Stack seed statistics
    stacked_seed_metrics = stack_seed_statistics(seed_metrics)
    stacked_seed_variables = stack_seed_statistics(seed_variables)

    # Compute mean and standard deviation
    seed_mean_std_metrics = compute_mean_and_std(stacked_seed_metrics)
    seed_mean_std_variables = compute_mean_and_std(stacked_seed_variables)

    # Save seed statistics
    metrics_path = args.output_path + 'metrics/'
    pathlib.Path(metrics_path).mkdir(parents=True, exist_ok=True)
    save_seed_statistics(seed_mean_std_metrics, metrics_path + args.nc_metrics + "_NC_metrics.pkl")
    save_seed_statistics(seed_mean_std_variables, metrics_path + args.nc_metrics + "_NC_variables.pkl")

    # Log seed statistics
    log_path = args.output_path + 'log/'
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    log_seed_statistics(seed_mean_std_metrics, log_path + args.nc_metrics + "_NC_metrics.txt")
    log_seed_statistics(seed_mean_std_variables, log_path + args.nc_metrics + "_NC_variables.txt")


def main():
    args = build_settings()
    aggregate_seed_statistics(args)


if __name__ == "__main__":
    main()
