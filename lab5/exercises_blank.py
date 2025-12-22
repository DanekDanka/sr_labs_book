# Exercises in order to perform laboratory work

# Import of modules
import numpy as np
from math import sqrt
import itertools
import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
from sklearn.metrics.pairwise import cosine_similarity
from common import get_eer


def get_tar_imp_scores(all_scores, all_labels):
    # Function to get target and impostors scores based on the labels

    tar_scores = []
    imp_scores = []
    for idx in range(len(all_labels)):

        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])

        else:
            imp_scores.append(all_scores[idx])

    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)

    return tar_scores, imp_scores

def plot_histograms_2sets(all_scores_1, all_labels_1,
                          all_scores_2, all_labels_2,
                          names=['in-domain', 'out-of-domain']):
    # Function to show target/impostor histograms and compute EER to out-of-domain and in-domain datasets
    
    # Get target and impostors scores
    tar_scores_1, imp_scores_1 = get_tar_imp_scores(all_scores_1, all_labels_1)
    tar_scores_2, imp_scores_2 = get_tar_imp_scores(all_scores_2, all_labels_2)

    # Plot histograms for target and impostor scores
    min_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).min()
    max_scores = np.concatenate((tar_scores_1, tar_scores_2,
                                 imp_scores_1, imp_scores_2)).max()

    hist(tar_scores_1, int(sqrt(len(tar_scores_1))), histtype='step', color='green',
         range=(min_scores, max_scores))
    hist(imp_scores_1, int(sqrt(len(imp_scores_1))), histtype='step', color='red',
         range=(min_scores, max_scores))
    hist(tar_scores_2, int(sqrt(len(tar_scores_2))), histtype='step', color='blue',
         range=(min_scores, max_scores))
    hist(imp_scores_2, int(sqrt(len(imp_scores_2))), histtype='step', color='cyan',
         range=(min_scores, max_scores))
    xlabel('$s$');
    ylabel('$\widehat{W}(s|H_0)$, $\widehat{W}(s|H_1)$');
    title('VoxCeleb1-O (cleaned), histograms');
    legend(list('{}_{}'.format(a[0], a[1]) for a in itertools.product(names, ['tar', 'imp'])))
    grid()
    show()

    # Compute equal error rate
    EER_1, thresh_EER_1 = get_eer(tar_scores_1, imp_scores_1)
    EER_2, thresh_EER_2 = get_eer(tar_scores_2, imp_scores_2)

    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[0], EER_1, thresh_EER_1))
    print("Equal Error Rate {0} (EER): {1:.3f}%, threshold EER: {2:.3f} ".format(names[1], EER_2, thresh_EER_2))

def mean_embd_norm(test_embds, adapt_embds):
    # Function to apply mean embedding normalization
    
    test_embds_adapted = {}
    adapt_embds_list = [adapt_embds[k] for k in adapt_embds.keys()]
    mean_embd = torch.stack(adapt_embds_list).mean(0)
    if len(mean_embd.size()) > 1:
        mean_embd = mean_embd.mean(0)

    for k in test_embds.keys():
        test_embds_adapted[k] = test_embds[k] - mean_embd
    
    return test_embds_adapted

def s_norm(test_data, lines, adapt_data, N_s=200, eps=0.5):
    """
    Function to perform s-normalization for scores with the snorm_data
    :param test_data: test embeddings
    :param lines: test protocol
    :param scores: raw scores matrix
    :param adapt_data: data for s-norm (s-norm embeddings)
    :param N_s: top N impostors scrores for s-normalization
    :param eps: epsilon for std
    :return: snorm_scores - s-normalized scores
    """
    
    scores_adapted = []
    all_labels = []
    all_trials = []

    # Prepare lists of unique wavs from protocols
    enroll_list = list(set(list([x.strip().split()[1] for x in lines])))
    test_list   = list(set(list([x.strip().split()[2] for x in lines])))
    adapt_list  = list(adapt_data.keys())

    # Prepare entolls: save enroll embds in ndarray [num_wavs x emb_size]
    E = []
    for id, enr in enumerate(enroll_list):
        E.append(test_data[enr].squeeze(0).numpy())
    E = np.array(E)

    # Prepare tests: save test embds in ndarray [num_wavs x emb_size]
    T = []
    for id, tst in enumerate(test_list):
        T.append(test_data[tst].squeeze(0).numpy())
    T = np.array(T)

    # Prepare adapt data: save adapt embds in ndarray [num_wavs x emb_size]
    A = []
    for id, a in enumerate(adapt_list):
        A.append(adapt_data[a].squeeze(0).numpy())
    A = np.array(A)
    
    ###########################################################
    # Here is your code
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create mapping from wav file to speaker ID and indices
    enroll_spk_map = {enroll_list[i]: enroll_list[i].split('/')[0] for i in range(len(enroll_list))}
    test_spk_map = {test_list[i]: test_list[i].split('/')[0] for i in range(len(test_list))}
    adapt_spk_map = {adapt_list[i]: adapt_list[i].split('/')[0] for i in range(len(adapt_list))}
    
    enroll_wav_to_idx = {enroll_list[i]: i for i in range(len(enroll_list))}
    test_wav_to_idx = {test_list[i]: i for i in range(len(test_list))}
    adapt_wav_to_idx = {adapt_list[i]: i for i in range(len(adapt_list))}
    
    E_torch = torch.FloatTensor(E).to(device)
    T_torch = torch.FloatTensor(T).to(device)
    A_torch = torch.FloatTensor(A).to(device)
    
    # Normalize embeddings for cosine similarity
    E_torch = torch.nn.functional.normalize(E_torch, p=2, dim=1)
    T_torch = torch.nn.functional.normalize(T_torch, p=2, dim=1)
    A_torch = torch.nn.functional.normalize(A_torch, p=2, dim=1)
    
    # Precompute all similarities: enroll vs adapt, test vs adapt
    # This is a large matrix multiplication: [num_enroll x emb_dim] @ [emb_dim x num_adapt] = [num_enroll x num_adapt]
    enroll_adapt_similarities = torch.mm(E_torch, A_torch.t()).cpu().numpy()  # [num_enroll x num_adapt]
    test_adapt_similarities = torch.mm(T_torch, A_torch.t()).cpu().numpy()    # [num_test x num_adapt]
    
    for line in tqdm.tqdm(lines, desc='Scoring with s-norm'):
        data = line.strip().split()
        label = int(data[0])
        enroll_wav = data[1]
        test_wav = data[2]
        
        enroll_spk_id = enroll_spk_map[enroll_wav]
        test_spk_id = test_spk_map[test_wav]
        
        enroll_idx = enroll_wav_to_idx[enroll_wav]
        test_idx = test_wav_to_idx[test_wav]
        
        # Compute raw score S using normalized embeddings
        enroll_emb = E_torch[enroll_idx:enroll_idx+1]
        test_emb = T_torch[test_idx:test_idx+1]
        S = torch.mm(enroll_emb, test_emb.t()).item()
        
        # Find impostor cohort indices (speakers that don't match enroll or test)
        impostor_indices = []
        for i, adapt_wav in enumerate(adapt_list):
            adapt_spk_id = adapt_spk_map[adapt_wav]
            if adapt_spk_id != enroll_spk_id and adapt_spk_id != test_spk_id:
                impostor_indices.append(i)
        
        if len(impostor_indices) == 0:
            scores_adapted.append(S)
            all_labels.append(label)
            all_trials.append(enroll_wav + " " + test_wav)
            continue
        
        impostor_indices = np.array(impostor_indices)
        
        # Get enroll scores with impostors (already computed)
        enroll_scores = enroll_adapt_similarities[enroll_idx, impostor_indices]
        enroll_scores_sorted = np.sort(enroll_scores)[::-1]  # Sort descending
        top_N_enroll = enroll_scores_sorted[:min(N_s, len(enroll_scores_sorted))]
        
        # Compute mean and std for enroll
        mu_e = np.mean(top_N_enroll)
        sigma_e = np.std(top_N_enroll)
        if sigma_e < eps:
            sigma_e = eps
        
        # Get test scores with impostors (already computed)
        test_scores = test_adapt_similarities[test_idx, impostor_indices]
        test_scores_sorted = np.sort(test_scores)[::-1]  # Sort descending
        top_N_test = test_scores_sorted[:min(N_s, len(test_scores_sorted))]
        
        # Compute mean and std for test
        mu_t = np.mean(top_N_test)
        sigma_t = np.std(top_N_test)
        if sigma_t < eps:
            sigma_t = eps
        
        S_norm = (S - mu_e) / (2 * sigma_e) + (S - mu_t) / (2 * sigma_t)
        
        scores_adapted.append(S_norm)
        all_labels.append(label)
        all_trials.append(enroll_wav + " " + test_wav)
    
    ###########################################################

    return scores_adapted, all_labels, all_trials

class CalibrationDataset(Dataset):

    def __init__(self, target_scores, impostor_scores):
        super(CalibrationDataset, self).__init__()

        self.target_scores   = target_scores
        self.impostor_scores = impostor_scores
        self.L_tar = len(target_scores)
        self.L_imp = len(impostor_scores)

    def __len__(self):
        
        return 1

    def __getitem__(self, idx):

        return self.target_scores, self.impostor_scores

class LinearCalibrationModel(torch.nn.Module):
    # Building of the full model for constructing the extractor of features
    
    def __init__(self):
        super(LinearCalibrationModel, self).__init__()
        
        self.calib_params = nn.Linear(1, 1)

    def forward(self, x):
        
        ###########################################################
        # Here is your code
        # Apply linear transformation: S_c = a*S + b
        calib_x = self.calib_params(x)
        ###########################################################

        return calib_x
    
class CalibrationLoss(nn.Module):

    def __init__(self, ptar=0.01):
        '''
        Calibration loss. Code is based on https://github.com/alumae/sv_score_calibration/blob/master/calibrate_scores.py
        :param ptar: probability of target hypothesis
        '''
        
        super(CalibrationLoss, self).__init__()
        
        self.ptar  = ptar
        self.alpha = np.log(ptar/(1 - ptar))

    def forward(self, target_llrs, nontarget_llrs):

        def negative_log_sigmoid(lodds):
            # Function to compute -log(sigmoid(log_odds))
            
            return torch.log1p(torch.exp(-lodds))
        
        loss_value = 0
        
        ###########################################################
        # Here is your code
        # Compute loss according to the formula:
        # J(a,b) = P(H_0)/N_0 * sum(ln(1 + exp(-(aS_i + b) + T_act^LLR))) 
        #        + (1 - P(H_0))/N_1 * sum(ln(1 + exp(aS_i + b - T_act^LLR)))
        # where T_act^LLR = alpha = ln(P(H_0)/(1 - P(H_0)))
        
        # Convert scores to log-odds (LLR)
        target_llrs_reshaped = target_llrs.view(-1, 1)
        nontarget_llrs_reshaped = nontarget_llrs.view(-1, 1)
        
        # Compute loss for target trials
        # ln(1 + exp(-(aS + b) + T_act^LLR)) = ln(1 + exp(-(LLR - T_act^LLR)))
        # Since target_llrs are already calibrated scores (aS + b), we need:
        # ln(1 + exp(-(target_llrs - self.alpha)))
        target_loss = negative_log_sigmoid(target_llrs_reshaped - self.alpha)
        target_loss = self.ptar * torch.mean(target_loss)
        
        # Compute loss for nontarget (impostor) trials
        # ln(1 + exp(aS + b - T_act^LLR)) = ln(1 + exp(LLR - T_act^LLR))
        # Since nontarget_llrs are already calibrated scores (aS + b), we need:
        # ln(1 + exp(nontarget_llrs - self.alpha))
        nontarget_loss = negative_log_sigmoid(-(nontarget_llrs_reshaped - self.alpha))
        nontarget_loss = (1 - self.ptar) * torch.mean(nontarget_loss)
        
        loss_value = target_loss + nontarget_loss
        ###########################################################

        return loss_value

def train_calibration(train_loader, model, criterion, optimizer, scheduler, num_epochs, verbose=False):
    # Function to train calibration model
    
    model.train()
    
    for epoch in range(0, num_epochs):
        
        for batch_idx, batch_data in enumerate(train_loader):
            tar_sc = batch_data[0]
            imp_sc = batch_data[1]
            
            ###########################################################
            # Here is your code
            tar_sc_tensor = torch.tensor(tar_sc, dtype=torch.float32).view(-1, 1)
            imp_sc_tensor = torch.tensor(imp_sc, dtype=torch.float32).view(-1, 1)

            tar_sc_calib = model(tar_sc_tensor)
            imp_sc_calib = model(imp_sc_tensor)
            
            loss = criterion(tar_sc_calib, imp_sc_calib)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ###########################################################
                            
        lr_value = optimizer.param_groups[0]['lr']

        if verbose:
            print("Epoch {:1.0f}, LR {:f} Loss {:f}".format(epoch, lr_value, loss.item()))
                
        scheduler[0].step()
    
    return