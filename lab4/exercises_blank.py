# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow
import matplotlib.pyplot as plt


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
    # Convert to numpy arrays to ensure proper indexing
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    sort_idx = np.argsort(all_scores)
    all_scores_sort = all_scores[sort_idx]
    ground_truth_sort = all_labels[sort_idx].astype(bool)
    
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
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        fnr_thr[idx] = np.sum(err[ground_truth_sort]) / len(tar_scores)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    
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
    
    plot(LLR, AC_thr, color='blue')
    xlabel('$LLR$'); ylabel('$\\bar{C}$'); title('Average cost'); grid(); show()
    
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
    len_P_Htar_thr = len(P_Htar_thr)
    tpr_thr = np.zeros(len_thr)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    tnr_thr = np.zeros(len_thr)
    AC = np.zeros([len_thr, len_P_Htar_thr])
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        ts = (solution == ground_truth_sort)                            # true solution vector
        err = (solution != ground_truth_sort)                          # error vector
        
        tpr_thr[idx] = np.sum(ts[ ground_truth_sort])/len(tar_scores)  # true positive ratio (TPR)
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        tnr_thr[idx] = np.sum(ts[ ~ground_truth_sort])/len(imp_scores) # true negative ratio (TNR)
        
        for idy in range(len_P_Htar_thr):
            AC[idx, idy] = C00*tpr_thr[idx]*P_Htar_thr[idy] + C10*fnr_thr[idx]*P_Htar_thr[idy] + \
                          C01*fpr_thr[idx]*(1 - P_Htar_thr[idy]) + C11*tnr_thr[idx]*(1 - P_Htar_thr[idy])  # Bayes' risk (average cost)
    
    # Plot average cost
    # Use safe indices
    start_idx = min(18705, len_thr - 200) if len_thr > 200 else 0
    end_idx = min(18905, len_thr)
    p_end_idx = min(999, len_P_Htar_thr - 1)
    
    if end_idx > start_idx and len_P_Htar_thr > 0:
        # Calculate aspect ratio to make the plot square
        x_range = P_Htar_thr[p_end_idx] - P_Htar_thr[0]
        y_range = LLR[start_idx] - LLR[end_idx-1]
        num_rows = end_idx - start_idx
        num_cols = len_P_Htar_thr
        
        if x_range > 0 and y_range > 0 and num_rows > 0 and num_cols > 0:
            # Aspect ratio: (y_range / num_rows) / (x_range / num_cols)
            aspect_ratio = (y_range * num_cols) / (x_range * num_rows)
            # Ensure aspect is finite and positive
            if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
                aspect_ratio = 'auto'
        else:
            aspect_ratio = 'auto'
        
        imshow(AC[start_idx:end_idx, :], extent=[P_Htar_thr[0], P_Htar_thr[p_end_idx], LLR[end_idx-1], LLR[start_idx]], aspect=aspect_ratio)
        xlabel('$P(H_0)$'); ylabel('$LLR$'); title('Average cost surface (top view)'); show()
    
    AC_P_Htar_max = np.zeros(len_thr)
    for idx in range(len_thr):
        AC_P_Htar_max[idx] = np.amax(AC[idx,:])
    AC_min_max_idx = np.argmin(AC_P_Htar_max)
    
    AC_thr_min = np.zeros(len_P_Htar_thr)
    for idy in range(len_P_Htar_thr):
        AC_thr_min[idy] = np.amin(AC[:, idy])
    AC_max_min_idx = np.argmax(AC_thr_min)
    
    solution = LLR > LLR[AC_min_max_idx]                               # decision
    err = (solution != ground_truth_sort)                               # error vector
    fnr = np.sum(err[ ground_truth_sort])/len(tar_scores)
    fpr = np.sum(err[~ground_truth_sort])/len(imp_scores)
    
    thr = LLR[AC_min_max_idx]
    AC = AC[AC_min_max_idx, AC_max_min_idx]
    P_Htar = P_Htar_thr[AC_max_min_idx]
    
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar