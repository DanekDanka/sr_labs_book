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
    for idx in range(len(all_labels)):
        if all_labels[idx] == 1:
            tar_scores.append(all_scores[idx])
        else:
            imp_scores.append(all_scores[idx])
    
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
    sort_idx = np.argsort(all_scores)
    all_scores_sort = all_scores[sort_idx]
    ground_truth_sort = np.array(all_labels)[sort_idx].astype(bool)
    
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    # Compute Gaussian PDF for target and impostor
    tar_gauss_pdf = (1.0 / (tar_scores_std * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((all_scores_sort - tar_scores_mean) / tar_scores_std) ** 2)
    imp_gauss_pdf = (1.0 / (imp_scores_std * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((all_scores_sort - imp_scores_mean) / imp_scores_std) ** 2)
    
    # Compute LLR = log(P(s|Htar) / P(s|Himp))
    LLR = np.log(tar_gauss_pdf + 1e-10) - np.log(imp_gauss_pdf + 1e-10)
    
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
    # Find threshold where FNR is closest to the given fnr
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]  # decision
        err = (solution != ground_truth_sort)  # error vector
        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)  # FNR
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)  # FPR
    
    # Find index where FNR is closest to the given fnr
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
    P_Himp = 1 - P_Htar
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    AC_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]  # decision
        err = (solution != ground_truth_sort)  # error vector
        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)  # FNR
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)  # FPR
        
        # Average cost: AC = C00*P(D0|H0)*P(H0) + C10*P(D1|H0)*P(H0) + C01*P(D0|H1)*P(H1) + C11*P(D1|H1)*P(H1)
        # P(D0|H0) = 1 - FPR, P(D1|H0) = FPR, P(D0|H1) = FNR, P(D1|H1) = 1 - FNR
        AC_thr[idx] = C00 * (1 - fpr_thr[idx]) * P_Himp + \
                     C10 * fpr_thr[idx] * P_Himp + \
                     C01 * fnr_thr[idx] * P_Htar + \
                     C11 * (1 - fnr_thr[idx]) * P_Htar
    
    # Find threshold that minimizes average cost
    AC_idx = np.argmin(AC_thr)
    thr = LLR[AC_idx]
    fnr = fnr_thr[AC_idx]
    fpr = fpr_thr[AC_idx]
    AC = AC_thr[AC_idx]
    
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
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    
    # Compute FNR and FPR for all thresholds
    for idx in range(len_thr):
        solution = LLR > LLR[idx]  # decision
        err = (solution != ground_truth_sort)  # error vector
        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)  # FNR
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)  # FPR
    
    # For minimax, find threshold that minimizes maximum cost over all P_Htar
    # We need to find the threshold where max cost is minimized
    max_AC_thr = np.zeros(len_thr)
    
    for idx in range(len_thr):
        # Compute cost for different P_Htar values and find maximum
        P_Htar_range = np.linspace(0.01, 0.99, 100)
        AC_range = np.zeros(len(P_Htar_range))
        
        for p_idx, P_Htar_val in enumerate(P_Htar_range):
            P_Himp_val = 1 - P_Htar_val
            AC_range[p_idx] = C00 * (1 - fpr_thr[idx]) * P_Himp_val + \
                             C10 * fpr_thr[idx] * P_Himp_val + \
                             C01 * fnr_thr[idx] * P_Htar_val + \
                             C11 * (1 - fnr_thr[idx]) * P_Htar_val
        
        max_AC_thr[idx] = np.max(AC_range)
    
    # Find threshold that minimizes maximum cost
    minmax_idx = np.argmin(max_AC_thr)
    thr = LLR[minmax_idx]
    fnr = fnr_thr[minmax_idx]
    fpr = fpr_thr[minmax_idx]
    AC = max_AC_thr[minmax_idx]
    
    # Find P_Htar that gives maximum cost at this threshold
    P_Htar_range = np.linspace(0.01, 0.99, 100)
    AC_range = np.zeros(len(P_Htar_range))
    for p_idx, P_Htar_val in enumerate(P_Htar_range):
        P_Himp_val = 1 - P_Htar_val
        AC_range[p_idx] = C00 * (1 - fpr) * P_Himp_val + \
                         C10 * fpr * P_Himp_val + \
                         C01 * fnr * P_Htar_val + \
                         C11 * (1 - fnr) * P_Htar_val
    
    P_Htar = P_Htar_range[np.argmax(AC_range)]
    
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar