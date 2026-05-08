import glob
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import random
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
warnings.filterwarnings('error')

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c.replace("_", "")) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l

def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({
        'tf': pd.Series(tpr - (1 - fpr), index=i),
        'threshold': pd.Series(threshold, index=i)
    })
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = roc_auc_score(y_test, pred_prob)
    
    evaluations = np.round((auc, sens, spec), 5)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['AUC', 'Sensitivity', 'Specificity']
    
    return evaluations

def get_avg_output(mydf, gt_col, pred_col, cutoff, nb_iters):
    idx_lst = list(range(len(mydf)))
    out_df = pd.DataFrame()
    
    for i in range(nb_iters):
        random.seed(i)
        bt_idx = [random.choice(idx_lst) for _ in range(len(idx_lst))]
        mydf_bt = mydf.iloc[bt_idx, :]
        tmpout_df = get_eval(mydf_bt[gt_col], mydf_bt[pred_col], cutoff)
        out_df = pd.concat([out_df, tmpout_df], axis=0)

    result_df = out_df.T
    result_df['Median'] = result_df.median(axis=1)
    result_df['LBD'] = result_df.quantile(0.025, axis=1)
    result_df['UBD'] = result_df.quantile(0.975, axis=1)

    output_lst = []
    for i in range(3):
        output_lst.append(
            '{:.3f}'.format(result_df['Median'].iloc[i]) + ' [' +
            '{:.3f}'.format(result_df['LBD'].iloc[i]) + ' - ' +
            '{:.3f}'.format(result_df['UBD'].iloc[i]) + ']'
        )

    result_df['output'] = output_lst
    myout = result_df.T
    
    return myout.iloc[-1, :]


dpath = "./Prediction/"

tgt_dir_lst = sort_nicely(glob.glob(os.path.join(dpath, 'Phewas_prediction/*.csv')))
out_dir = os.path.join(dpath, "Phewas_prediction/Evaluation")
os.makedirs(out_dir, exist_ok=True)


for tgt_dir in tqdm(tgt_dir_lst):
    tgt = os.path.basename(tgt_dir)[:-4]
    tgt_pred_df = pd.read_csv(tgt_dir)

    ct_idp = Find_Optimal_Cutoff(tgt_pred_df["target_y"], tgt_pred_df["y_pred_idp"])[0]
    ct_cov = Find_Optimal_Cutoff(tgt_pred_df["target_y"], tgt_pred_df["y_pred_cov"])[0]
    ct_idp_cov = Find_Optimal_Cutoff(tgt_pred_df["target_y"], tgt_pred_df["y_pred_idp_cov"])[0]

    res_idp = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_idp', ct_idp, nb_iters=1000)
    res_cov = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_cov', ct_cov, nb_iters=1000)
    res_idp_cov = get_avg_output(tgt_pred_df, 'target_y', 'y_pred_idp_cov', ct_idp_cov, nb_iters=1000)

    res_df = pd.concat([res_idp, res_cov, res_idp_cov], axis=1)
    res_df = res_df.T
    res_df.index = ['IDP', 'Covariates', 'IDP+Covariates']

    res_df.to_csv(os.path.join(out_dir, tgt + ".csv"), index=True)





