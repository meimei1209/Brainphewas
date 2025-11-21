import sys
import subprocess

# ----------------------------------------------------------
# Auto-install dependencies if missing
# ----------------------------------------------------------
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package,
                           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
for pkg in ["lightgbm", "joblib", "tqdm", "scikit-learn", "pandas", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import os
import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# Global parameters
# ----------------------------------------------------------
nb_cpus = 6
nb_params = 50
my_seed = 2024
random.seed(my_seed)
top_k = 30   

my_params0 = {
    "n_estimators": 200,
    "max_depth": 10,
    "num_leaves": 20,
    "subsample": 0.8,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8
}

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------
def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    if mysum == 0:
        return mydict
    return {k: v / mysum for k, v in mydict.items()}

def get_cov_f_lst(tgt2pred_df, tgt, full_idp_f_lst):
    row = tgt2pred_df.loc[tgt2pred_df.Disease_code == tgt, "SEX"]
    sex_id = 0 if row.empty else row.iloc[0]
    if sex_id == 1:
        cov_f_lst = ["age", "ethnicity", "BMI", "education", "tdi",
                     "smoking_status", "drinking_status", "Total_IntraCranial_ins2",
                      "PRS_total_triglyceride"]
    else:
        cov_f_lst = ["age", "sex", "ethnicity", "BMI", "education", "tdi",
                     "smoking_status", "drinking_status", "Total_IntraCranial_ins2",
                     "PRS_total_triglyceride"]
    if any(f.startswith("edges") or f.startswith("nodes") for f in full_idp_f_lst):
        cov_f_lst += ["rfMRI_head_motion_ins2", "rfMRI_signal_to_noise_ins2"]
    return cov_f_lst

def get_idp_f_lst(mydf, train_idx, f_lst, my_params, top_k=30):
    X_train, y_train = mydf.iloc[train_idx][f_lst], mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc',
                            is_unbalance=True, seed=my_seed, n_jobs=1)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    tg_imp_df = pd.DataFrame({'Feature': my_lgb.booster_.feature_name(),
                              'TotalGain': totalgain_imp})
    tg_imp_df.sort_values(by='TotalGain', inplace=True, ascending=False)
    return tg_imp_df.Feature.tolist()[:top_k]

def get_best_params(mydf, idp_f_lst, my_params_lst):
    kf = KFold(n_splits=5, shuffle=True, random_state=my_seed)
    results = []
    for my_params in my_params_lst:
        auc_cv_lst = []
        for train_idx, test_idx in kf.split(mydf):
            X_train, y_train = mydf.iloc[train_idx][idp_f_lst], mydf.iloc[train_idx].target_y
            X_test, y_test = mydf.iloc[test_idx][idp_f_lst], mydf.iloc[test_idx].target_y
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            my_lgb = LGBMClassifier(objective='binary', metric='auc',
                                    is_unbalance=True, seed=my_seed, n_jobs=1)
            my_lgb.set_params(**my_params)
            my_lgb.fit(X_train, y_train)
            y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred_prob))
        if auc_cv_lst:
            results.append((np.mean(auc_cv_lst), my_params))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1] if results else my_params0

def get_best_params_with_score(mydf, idp_f_lst, my_params_lst):
    kf = KFold(n_splits=5, shuffle=True, random_state=my_seed)
    best_mean, best_se, best_params = -1.0, None, None
    for my_params in my_params_lst:
        auc_cv_lst = []
        for train_idx, test_idx in kf.split(mydf):
            X_train = mydf.iloc[train_idx][idp_f_lst]
            y_train = mydf.iloc[train_idx].target_y
            X_test  = mydf.iloc[test_idx][idp_f_lst]
            y_test  = mydf.iloc[test_idx].target_y
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            clf = LGBMClassifier(objective='binary', metric='auc',
                                 is_unbalance=True, seed=my_seed, n_jobs=1)
            clf.set_params(**my_params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            auc_cv_lst.append(roc_auc_score(y_test, y_pred))
        if auc_cv_lst:
            mean_auc = float(np.mean(auc_cv_lst))
            se_auc = float(np.std(auc_cv_lst, ddof=1) / np.sqrt(len(auc_cv_lst)))
            if mean_auc > best_mean:
                best_mean, best_se, best_params = mean_auc, se_auc, my_params
    if best_params is None:
        return my_params0, 0.5, 0.0
    return best_params, best_mean, best_se


def model_training(mydf, train_idx, test_idx, f_lst, my_params):
    X_train, X_test = mydf.iloc[train_idx][f_lst], mydf.iloc[test_idx][f_lst]
    y_train = mydf.iloc[train_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc',
                            is_unbalance=True, seed=my_seed, n_jobs=1)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred = my_lgb.predict_proba(X_test)[:, 1].tolist()
    return y_pred, my_lgb

def get_iter_predictions(mydf, full_idp_f_lst, cov_f_lst, fold_id, my_params0, my_params, top_k=30):
    test_idx = np.where(mydf["UK_assessment_centre_ins2"].values == fold_id)[0].tolist()
    train_idx = np.where(mydf["UK_assessment_centre_ins2"].values != fold_id)[0].tolist()
    if (len(test_idx) == 0) or (len(train_idx) == 0):
        return (Counter(), Counter(), [], [], [], [], [], None) 
    if sum(mydf.target_y.iloc[test_idx]) == 0:
        return (Counter(), Counter(), [], [], [], [], [], None) 

    full_order = get_idp_f_lst(mydf, train_idx, full_idp_f_lst, my_params0, top_k=len(full_idp_f_lst))

    top_k_candidates = [10, 20, 30, 40]

    k_records = []  
    for K in top_k_candidates:
        prefix = full_order[:K]
        params_K, mean_auc_K, se_auc_K = get_best_params_with_score(mydf, prefix, my_params)
        k_records.append((K, params_K, mean_auc_K, se_auc_K))

    max_mean = max(r[2] for r in k_records)
    max_se = next(r[3] for r in k_records if r[2] == max_mean)
    cutoff = max_mean - max_se
    feasible = [r for r in k_records if r[2] >= cutoff]
    selected = min(feasible, key=lambda x: x[0]) if feasible else max(k_records, key=lambda x: x[2])

    selected_k, my_params_idp, _, _ = selected
    idp_f_lst = full_order[:selected_k]

    my_params_cov = get_best_params(mydf, cov_f_lst, my_params)
    my_params_idp_cov = get_best_params(mydf, cov_f_lst + idp_f_lst, my_params)

    y_pred_idp, lgb_idp = model_training(mydf, train_idx, test_idx, idp_f_lst, my_params_idp)
    y_pred_cov, lgb_cov = model_training(mydf, train_idx, test_idx, cov_f_lst, my_params_cov)
    y_pred_idp_cov, lgb_idp_cov = model_training(mydf, train_idx, test_idx, cov_f_lst + idp_f_lst, my_params_idp_cov)

    y_test_lst = mydf.target_y.iloc[test_idx].tolist()
    eid_lst = mydf.eid.iloc[test_idx].tolist()

    totalgain_imp = lgb_idp.booster_.feature_importance(importance_type="gain")
    totalcover_imp = lgb_idp.booster_.feature_importance(importance_type="split")
    tg_imp_cv = Counter(normal_imp(dict(zip(lgb_idp.booster_.feature_name(), totalgain_imp.tolist()))))
    tc_imp_cv = Counter(normal_imp(dict(zip(lgb_idp.booster_.feature_name(), totalcover_imp.tolist()))))

    return (tg_imp_cv, tc_imp_cv, eid_lst, y_test_lst, y_pred_idp, y_pred_cov, y_pred_idp_cov, selected_k)

# ----------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------
def main():
    import dxpy

    dpath = "/opt/notebooks/"

    tgt2pred_df = pd.read_csv(os.path.join(dpath, 'Incident.csv'), dtype={'Disease_code': str})
    idp_df = pd.read_csv(os.path.join(dpath, 'ImagingData.csv'))
    cov_needed = [
    "eid", "age", "sex", "ethnicity", "BMI", "education", "tdi",
    "smoking_status", "drinking_status", "Total_IntraCranial_ins2",
    "PRS_total_triglyceride",
    "UK_assessment_centre_ins2", "rfMRI_head_motion_ins2", "rfMRI_signal_to_noise_ins2"
    ]
    cov_df = pd.read_csv(os.path.join(dpath, 'Full_Covariates.csv'), usecols=cov_needed)

    modality_tag = "All"

    cat_cols = ["sex", "ethnicity", "BMI", "education",
                "smoking_status", "drinking_status", "UK_assessment_centre_ins2"]
    for col in cat_cols:
        if col in cov_df.columns:
            cov_df[col] = cov_df[col].astype("category")
    for col in ["ethnicity", "BMI", "smoking_status", "drinking_status"]:
        if col in cov_df.columns:
            cov_df[col] = cov_df[col].cat.add_categories("Unknown").fillna("Unknown")
    for col in ["tdi", "rfMRI_head_motion_ins2", "rfMRI_signal_to_noise_ins2", "Total_IntraCranial_ins2",
                "PRS_total_triglyceride"]:
        if col in cov_df.columns:
            cov_df[col] = cov_df[col].fillna(cov_df[col].median())

    full_idp_f_lst = idp_df.columns.tolist()[1:]
    mydf = pd.merge(cov_df, idp_df, how='inner', on=['eid']).dropna().reset_index(drop=True)

    params_dict = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": np.linspace(5, 30, 6).astype("int32").tolist(),
        "num_leaves": np.linspace(5, 30, 6).astype("int32").tolist(),
        "subsample": np.linspace(0.6, 1, 9).tolist(),
        "learning_rate": [0.1, 0.05, 0.01, 0.001],
        "colsample_bytree": np.linspace(0.6, 1, 9).tolist()
    }
    candidate_params_lst = select_params_combo(params_dict, nb_params, my_seed)

    all_tgts = ["272.1"]

    auc_summary = []

    for tgt in tqdm(all_tgts, desc=f"Disease-loop (autoK)", ncols=100):
        try:
            cov_f_lst = get_cov_f_lst(tgt2pred_df, tgt, full_idp_f_lst)
            file_path = os.path.join(dpath, 'Targets2Analysis', f"{tgt}.csv")
            if not os.path.exists(file_path):
                continue
            tmp_tgt_df = pd.read_csv(file_path, usecols=['eid', 'target_y', 'BL2Target_yrs'])
            tmp_tgt_df = tmp_tgt_df.loc[tmp_tgt_df.BL2Target_yrs > 0].reset_index(drop=True)
            tmp_df = pd.merge(tmp_tgt_df, mydf, how='inner', on=['eid']).dropna().reset_index(drop=True)

            eid_lst, y_test_lst, y_pred_idp_lst, y_pred_cov_lst, y_pred_idp_cov_lst = [], [], [], [], []
            tg_imp_cv, tc_imp_cv = Counter(), Counter()

            fold_id_lst_tmp = sorted(tmp_df["UK_assessment_centre_ins2"].unique())
            fold_results_lst = Parallel(n_jobs=nb_cpus)(
                delayed(get_iter_predictions)(
                    tmp_df, full_idp_f_lst, cov_f_lst, fold_id, my_params0, candidate_params_lst, top_k=top_k
                )
                for fold_id in fold_id_lst_tmp
            )

            selected_k_list = []

            for fold_results in fold_results_lst:
                tg_imp_cv += fold_results[0]
                tc_imp_cv += fold_results[1]
                eid_lst += fold_results[2]
                y_test_lst += fold_results[3]
                y_pred_idp_lst += fold_results[4]
                y_pred_cov_lst += fold_results[5]
                y_pred_idp_cov_lst += fold_results[6]
                if len(fold_results) >= 8 and fold_results[7] is not None:
                    selected_k_list.append(fold_results[7])

            tg_imp_cv = normal_imp(tg_imp_cv)
            tc_imp_cv = normal_imp(tc_imp_cv)
            imp_df = pd.merge(
                pd.DataFrame({'Feature': list(tc_imp_cv.keys()), 'TotalCover_cv': list(tc_imp_cv.values())}),
                pd.DataFrame({'Feature': list(tg_imp_cv.keys()), 'TotalGain_cv': list(tg_imp_cv.values())}),
                on='Feature', how='left'
            )
            imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)

            final_top_k = int(np.median(selected_k_list)) if selected_k_list else top_k

            imp_path = os.path.join(dpath, f"Results/FeatureImportance/{modality_tag}_{tgt}_Top{final_top_k}.csv")
            os.makedirs(os.path.dirname(imp_path), exist_ok=True)
            imp_df.to_csv(imp_path, index=False)
            subprocess.run(["dx", "upload", imp_path,
                            "--path", f"/Prediction_model/Results/FeatureImportance/Specific_{modality_tag}_{tgt}_Top{final_top_k}.csv"], check=True)

            auc_idp, auc_cov, auc_idp_cov = None, None, None
            if len(set(y_test_lst)) > 1:
                auc_idp = roc_auc_score(y_test_lst, y_pred_idp_lst)
                auc_cov = roc_auc_score(y_test_lst, y_pred_cov_lst)
                auc_idp_cov = roc_auc_score(y_test_lst, y_pred_idp_cov_lst)

            pred_df = pd.DataFrame({
                'eid': eid_lst,
                'target_y': y_test_lst,
                'y_pred_idp': y_pred_idp_lst,
                'y_pred_cov': y_pred_cov_lst,
                'y_pred_idp_cov': y_pred_idp_cov_lst
            })
            pred_path = os.path.join(dpath, f"Results/Predictions/{modality_tag}_{tgt}_Top{final_top_k}.csv")
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            pred_df.to_csv(pred_path, index=False)
            subprocess.run(["dx", "upload", pred_path,
                            "--path", f"/Prediction_model/Results/Predictions/Specific_{modality_tag}_{tgt}_Top{final_top_k}.csv"], check=True)

            auc_summary.append({
                "Disease": tgt,
                "Cases": sum(tmp_df["target_y"]),
                "N": len(tmp_df),
                "TopK": final_top_k, 
                "AUC_idp": auc_idp,
                "AUC_cov": auc_cov,
                "AUC_idp_cov": auc_idp_cov,
                "Fold_TopKs": ",".join(map(str, selected_k_list)) if selected_k_list else ""
            })

        except Exception as e:
            print(f"❌ Error in {tgt}: {e}")


    auc_df = pd.DataFrame(auc_summary)
    auc_dir = os.path.join(dpath, "Results/AUC")
    os.makedirs(auc_dir, exist_ok=True)
    auc_path = os.path.join(auc_dir, f"{modality_tag}_AUC_summary_autoK.csv")
    auc_df.to_csv(auc_path, index=False)
    subprocess.run(["dx", "upload", auc_path,
                    "--path", f"/Prediction_model/Results/AUC/Specific_{modality_tag}_AUC_summary_autoK.csv"], check=True)

if __name__ == "__main__":
    main()
