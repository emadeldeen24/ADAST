import torch
import os
from utils import fix_randomness, _logger, calc_metrics_all_runs, copy_Files
from dataloader.dataloader import data_generator
from trainer.ADAST import cross_domain_train
from trainer.training_evaluation import cross_domain_test
from datetime import datetime
from config_files.configs import Config as Configs
import argparse

######## ARGS ######################
parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default='experiments_logs', type=str,
                    help='Directory used to save experimental results')

parser.add_argument('--experiment_description', default='tests', type=str,
                    help='Main experiment Description')

parser.add_argument('--run_description', default='ADAST_test', type=str,
                    help='Each experiment may have multiple runs, with specific setting in each:')

# Domain adaptation method / Dataset / Model
parser.add_argument('--da_method', default='ADAST', type=str,
                    help='method selection')

# Experiment setting
parser.add_argument('--num_runs', default=1, type=int,
                    help='Number of consecutive run with different seeds')

parser.add_argument('--device', default='cuda:3', type=str,
                    help='cpu or cuda')

home_dir = os.getcwd()
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--plot_umap', default=False, type=bool,
                    help='Plot UMAP for training and testing or not?')
args = parser.parse_args()

###################################


start_time = datetime.now()
device = torch.device(args.device)
da_method = args.da_method
save_dir = args.save_dir
configs = Configs()
data_path = f"./data"

experiment_description = args.experiment_description

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

seeds = range(10)


def main_train_cd():
    # find out the domains IDs
    data_files = os.listdir(data_path)
    x_domains = [("a", "b"), ("a", "c"), ("b", "a"), ("b", "c"), ("c", "a"), ("c", "b")]

    # Logging
    exp_log_dir = os.path.join(save_dir, experiment_description, args.run_description)
    os.makedirs(exp_log_dir, exist_ok=True)

    # save a copy of training files:
    copy_Files(exp_log_dir, da_method)

    # loop through domains
    for i in x_domains:
        src_id = i[0]
        trg_id = i[1]

        # specify number of consecutive runs
        for run_id in range(args.num_runs):
            fix_randomness(seeds[run_id])

            # Logging
            log_dir = os.path.join(exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))
            os.makedirs(log_dir, exist_ok=True)
            log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
            logger = _logger(log_file_name)
            logger.debug("=" * 45)
            logger.debug(f'Method:  {da_method}')
            logger.debug("=" * 45)
            logger.debug(f'Source: {src_id} ---> Target: {trg_id}')
            logger.debug(f'Run ID: {run_id}')
            logger.debug("=" * 45)

            # Load datasets
            src_train_dl, src_valid_dl, src_test_dl = data_generator(data_path, src_id, configs)
            trg_train_dl, trg_valid_dl, trg_test_dl = data_generator(data_path, trg_id, configs)

            target_model = cross_domain_train(src_train_dl, trg_train_dl, trg_valid_dl,
                                                            src_id, trg_id,
                                                            device, logger, configs, args, configs.adast_params)
            # to test the model and generate results ...
            cross_domain_test(target_model, src_test_dl, trg_test_dl,
                              device, log_dir, logger, args)

    calc_metrics_all_runs(args.save_dir, experiment_description, args.run_description,
                          (args.da_method, experiment_description, args.run_description))
    logger.debug(f"Running time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main_train_cd()
