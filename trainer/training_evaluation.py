import os
import torch
import torch.nn as nn
import numpy as np
from utils import _calc_metrics, _plot_umap


def val_self_training(model,valid_dl,  device, src_id, trg_id, round_idx, args):
    from sklearn.metrics import accuracy_score
    model[0].eval()
    model[1].eval()
    model[2][0].eval()
    model[2][1].eval()

    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])    
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for data, labels in valid_dl:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            out = model[0](data)
            features = model[1](out)
            predictions = model[2][0](features)
            predictions2 = model[2][1](features)
            
            predictions = torch.mean(torch.stack([predictions, predictions2,]), dim=0)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())
            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    # print("agreement of labels: ", accuracy_score(all_labels, all_pseudo_labels))
    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_{src_id}_to_{trg_id}_round_{round_idx}.pt"
    os.makedirs(os.path.join(args.home_path ,"data"), exist_ok =True)
    torch.save(data_save, os.path.join(args.home_path ,"data", file_name))




def model_evaluate(model, valid_dl, device):
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2][0].eval()
        model[2][1].eval()

    else:
        model.eval()
    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])


    with torch.no_grad():
        for data, labels in valid_dl:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            out = model[0](data)

            features = model[1](out)
            predictions = model[2][0](features)
            predictions2 = model[2][1](features)

            predictions = torch.max(predictions, predictions2)

            # compute loss
            loss = criterion(predictions, labels)
            total_loss.append(loss.item())
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())



    total_loss = torch.tensor(total_loss).mean() # average loss
    total_acc = torch.tensor(total_acc).mean()   #average acc
    return total_loss, total_acc, outs, trgs


def cross_domain_test(target_model, src_test_dl, tgt_test_dl,
                      device, log_dir,logger, args):

    # finish Training evaluate on test sets
    logger.debug('==== Domain Adaptation completed =====')
    logger.debug('\n==== Evaluate on test sets ===========')
    
    target_loss, target_score, pred_labels, true_labels = model_evaluate(target_model, tgt_test_dl, device)
    torch.save(target_model, os.path.join(log_dir, "model.pt"))
    
    _calc_metrics(pred_labels, true_labels, log_dir, args.home_path)

    logger.debug(f'\t {args.da_method} Loss     : {target_loss:.4f}\t | \t{args.da_method} Accuracy     : {target_score:2.4f}')

    if args.plot_umap:
        _plot_umap(target_model, src_test_dl, tgt_test_dl, device, log_dir, f'{args.da_method}', 'test')


