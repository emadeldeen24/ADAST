import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score
import os
import sys
import logging
from shutil import copy
from collections import OrderedDict

import matplotlib.pyplot as plt


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark  = False


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True) 
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    file_name = os.path.basename(os.path.normpath(log_dir)) + "_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)



def calc_metrics_all_runs(save_dir, experiment_dir, run_dir, da_method):
    base_dir = os.path.join(save_dir, experiment_dir, run_dir)

    odict = OrderedDict()

    runs_dirs = []
    for i in os.listdir(base_dir):
        runs_dirs.append(i)

    runs_dirs = [os.path.join(i) for i in runs_dirs if "run_" in i]

    # result dataframe
    column_names = ['Scenario', 'Acc-mean', "Acc-std", 'MF1-mean', "MF1-std"]
    df = pd.DataFrame(columns=column_names)

    for i in runs_dirs:
        pred_labels = np.load(os.path.join(base_dir, i, "labels", "predicted_labels.npy"))
        true_labels = np.load(os.path.join(base_dir, i, "labels", "true_labels.npy"))

        pred_labels = np.array(pred_labels).astype(int)
        true_labels = np.array(true_labels).astype(int)

        r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
        ACC = accuracy_score(true_labels, pred_labels)
        scenario_parts = i.split("_")

        if scenario_parts[0] + "-->" + scenario_parts[2] not in odict:
            odict[scenario_parts[0] + "-->" + scenario_parts[2]] = dict()
        odict[scenario_parts[0] + "-->" + scenario_parts[2]][scenario_parts[-1]] = [ACC, r["macro avg"]["f1-score"]]


    for counter, i in enumerate(list(odict.keys())):
        values = np.array(list(odict[i].values()))

        df.loc[counter] = [i, values.mean(0)[0], values.std(0)[0], values.mean(0)[1], values.std(0)[1]]

    # Add averages
    avg_acc = df["Acc-mean"].mean()
    avg_f1 = df["MF1-mean"].mean()
    df.loc[counter + 1] = ["Average", df["Acc-mean"].mean(), df["Acc-std"].mean(), df["MF1-mean"].mean(),
                           df["MF1-std"].mean()]

    df["Acc-mean"] = df["Acc-mean"] * 100
    df["Acc-std"] = df["Acc-std"] * 100
    df["MF1-mean"] = df["MF1-mean"] * 100
    df["MF1-std"] = df["MF1-std"] * 100

    # save classification report
    file_name = f"{da_method[0]}_{da_method[1]}_{da_method[2]}.xlsx"
    report_Save_path = os.path.join(base_dir, file_name)
    df.to_excel(report_Save_path)
    return avg_acc, avg_f1


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, da_method):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("train_CD.py", os.path.join(destination_dir, "train_CD.py"))
    copy(f"trainer/{da_method}.py", os.path.join(destination_dir, f"{da_method}.py"))
    copy(f"trainer/training_evaluation.py", os.path.join(destination_dir, f"training_evaluation.py"))
    copy(f"config_files/configs.py", os.path.join(destination_dir, f"configs.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))


def _plot_umap(model, src_dl, trg_dl, device, save_dir, model_type,
               train_mode):  # , layer_output_to_plot, y_test, save_dir, type_id):
    import umap
    import umap.plot
    classes_names = ['W', 'N1', 'N2', 'N3', 'REM']
    
    font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 17}
    plt.rc('font', **font)
    
    with torch.no_grad():
        # Source flow
        src_data = src_dl.dataset.x_data.float().to(device)
        src_labels = src_dl.dataset.y_data.view((-1)).long()
        out = model[0](src_data)
        src_features = model[1](out)

        # target flow
        trg_data = trg_dl.dataset.x_data.float().to(device)
        trg_labels = trg_dl.dataset.y_data.view((-1)).long()
        out = model[0](trg_data)
        trg_features = model[1](out)

    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))
        
    #cmaps = plt.get_cmap('jet')
    src_model_reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, metric='correlation', random_state=42)
    src_embedding = src_model_reducer.fit_transform(src_features.detach().cpu().numpy())
    
    trg_model_reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, metric='correlation', random_state=42)
    trg_embedding = src_model_reducer.fit_transform(trg_features.detach().cpu().numpy())
    
    print("Plotting UMAP for " + model_type + "...")
    plt.figure(figsize=(16, 10))
    src_scatter = plt.scatter(src_embedding[:, 0], src_embedding[:, 1], c=src_labels,  s=10, label="Source", marker='o')
    trg_scatter = plt.scatter(trg_embedding[:, 0], trg_embedding[:, 1], c=trg_labels, s=10, label="Target", marker='x', alpha=0.4)
    handles, _ = src_scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names,  title="Classes")
    file_name = "umap_" + model_type + "_" + train_mode + ".png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()
    
    print("Plotting UMAP for domain-based " + model_type + "...")
    plt.figure(figsize=(16, 10))
    plt.scatter(src_embedding[:, 0], src_embedding[:, 1], s=10, c='red', label="Source")
    plt.scatter(trg_embedding[:, 0], trg_embedding[:, 1], s=10, c='blue', label="Target")
    plt.legend()
    file_name = "umap_" + model_type + "_" + train_mode + "_domain-based.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()

    
    print("Plotting UMAP for target domain " + model_type + "...")
    plt.figure(figsize=(16, 10))
    plt.scatter(trg_embedding[:, 0], trg_embedding[:, 1], s=10, c=trg_labels, label="Target")
    handles, _ = src_scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names,  title="Classes")
    file_name = "umap_" + model_type + "_" + train_mode + "_target_domain.png"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()



def get_model_params(net):
    """Get parameters of models by name."""
    for n, p in net.named_parameters():
        if p.requires_grad:
            return p
            
def calc_similiar_penalty(classifier_1, classifier_2):
    """Calculate similiar penalty |W_1^T W_2|."""
    clf_1_params = get_model_params(classifier_1)
    clf_2_params = get_model_params(classifier_2)

    similiar_penalty = torch.sum(
        torch.abs(torch.mm(clf_1_params.transpose(0, 1), clf_2_params)))
    return similiar_penalty