# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for score, label in zip(all_scores, all_labels):
        if label == 1:
            tar_scores.append(score)
        else:
            imp_scores.append(score)
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    sorted_indices = np.argsort(all_scores)
    all_scores_sort = all_scores[sorted_indices]
    ground_truth_sort = all_labels[sorted_indices].astype(bool)
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    for i, score in enumerate(all_scores_sort):
        tar_gauss_pdf[i] = gauss_pdf(score, tar_scores_mean, tar_scores_std)
        imp_gauss_pdf[i] = gauss_pdf(score, imp_scores_mean, imp_scores_std)
        LLR[i] = np.log(tar_gauss_pdf[i] / (imp_gauss_pdf[i] + 1e-10))
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    
    fnr_idx = np.argmin(np.abs(fnr_thr - fnr))
    thr = LLR[fnr_idx]
    fpr = fpr_thr[fnr_idx]
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    thr = np.log(((C01 - C11) * (1 - P_Htar)) / ((C10 - C00) * P_Htar + 1e-10))
    
    solution = LLR > thr
    err = (solution != ground_truth_sort)
    fnr = np.sum(err[ground_truth_sort]) / len(tar_scores)
    fpr = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    
    P_Himp = 1 - P_Htar
    AC = (C00 * (1 - fnr) * P_Htar + 
          C10 * fnr * P_Htar + 
          C01 * fpr * P_Himp + 
          C11 * (1 - fpr) * P_Himp)
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    AC_values = np.zeros(len(P_Htar_thr))
    
    for i, P_Htar_val in enumerate(P_Htar_thr):
        thr_temp = np.log(((C01 - C11) * (1 - P_Htar_val)) / ((C10 - C00) * P_Htar_val + 1e-10))
        
        solution = LLR > thr_temp
        err = (solution != ground_truth_sort)
        fnr_temp = np.sum(err[ground_truth_sort]) / len(tar_scores)
        fpr_temp = np.sum(err[~ground_truth_sort]) / len(imp_scores)
        
        AC_values[i] = (C00 * (1 - fnr_temp) * P_Htar_val + 
                       C10 * fnr_temp * P_Htar_val + 
                       C01 * fpr_temp * (1 - P_Htar_val) + 
                       C11 * (1 - fpr_temp) * (1 - P_Htar_val))
    
    max_idx = np.argmax(AC_values)
    P_Htar = P_Htar_thr[max_idx]
    AC = AC_values[max_idx]
    
    thr = np.log(((C01 - C11) * (1 - P_Htar)) / ((C10 - C00) * P_Htar + 1e-10))
    solution = LLR > thr
    err = (solution != ground_truth_sort)
    fnr = np.sum(err[ground_truth_sort]) / len(tar_scores)
    fpr = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar