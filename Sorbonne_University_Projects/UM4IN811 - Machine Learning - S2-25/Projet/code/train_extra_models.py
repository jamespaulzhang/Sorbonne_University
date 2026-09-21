
#Entraîne des modèles supplémentaires en réutilisant

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import pickle
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src import config
from src.models import macro_auc_skip_empty


data = np.load(config.CACHE_DIR / "features_full.npz")
X = data['X']
valid_mask = data['valid_mask']

train_idx = np.load(config.MODELS_DIR / "train_idx.npy")
val_idx = np.load(config.MODELS_DIR / "val_idx.npy")
Y_val = np.load(config.MODELS_DIR / "Y_val.npy")


import pandas as pd
import ast
train_df = pd.read_csv(config.DATA_DIR / "train.csv")
sub = pd.read_csv(config.DATA_DIR / "sample_submission.csv")
species_list = sub.columns[1:].tolist()
sp2idx = {sp: i for i, sp in enumerate(species_list)}

Y_all = np.zeros((len(train_df), len(species_list)), dtype=np.float32)
for r_idx, row in enumerate(train_df.itertuples()):
    if row.primary_label in sp2idx:
        Y_all[r_idx, sp2idx[row.primary_label]] = 1.0
    try:
        sec = ast.literal_eval(row.secondary_labels) if isinstance(row.secondary_labels, str) else []
        for s in sec:
            if s in sp2idx:
                Y_all[r_idx, sp2idx[s]] = 1.0
    except (ValueError, SyntaxError):
        pass
Y = Y_all[valid_mask]

X_tr, X_val = X[train_idx], X[val_idx]
Y_tr = Y[train_idx]

print(f"X_tr: {X_tr.shape}, X_val: {X_val.shape}, Y_tr: {Y_tr.shape}")
print(f"Vérification Y_val identique : {np.array_equal(Y_val, Y[val_idx])}\n")

n_species = len(species_list)



# Fonction d'entraînement Binary Relevance

def train_br(make_model, X_tr, Y_tr, X_val, name,
             needs_scaling=False, min_pos=3):
    """
    Entraîne un classifieur binaire par espèce et renvoie les prédictions
    sur le val set. make_model() doit renvoyer une instance fraîche du modèle.
    """
    print(f"=== {name} ===")
    t0 = time.time()
    
    if needs_scaling:
        scaler = StandardScaler().fit(X_tr)
        X_tr_use = np.nan_to_num(scaler.transform(X_tr))
        X_val_use = np.nan_to_num(scaler.transform(X_val))
    else:
        scaler = None
        X_tr_use, X_val_use = X_tr, X_val
    
    models = {}
    val_preds = np.zeros((len(X_val), n_species), dtype=np.float32)
    
    for i, sp in enumerate(species_list):
        y_sp = Y_tr[:, i]
        n_pos = int(y_sp.sum())
        if n_pos < min_pos or n_pos == len(y_sp):
            models[sp] = None
            continue
        try:
            m = make_model()
            m.fit(X_tr_use, y_sp)
            models[sp] = m
            # predict_proba si dispo, sinon decision_function normalisé
            if hasattr(m, 'predict_proba'):
                val_preds[:, i] = m.predict_proba(X_val_use)[:, 1]
            else:
                val_preds[:, i] = m.decision_function(X_val_use)
        except Exception as e:
            models[sp] = None
        if (i + 1) % 50 == 0:
            print(f"  {name} : {i+1}/{n_species}")
    
    auc, n_eval = macro_auc_skip_empty(Y_val, val_preds)
    elapsed = (time.time() - t0) / 60
    print(f"{name} terminé en {elapsed:.1f} min — macro AUC = {auc:.4f} ({n_eval} espèces)\n")
    
    # Sauvegarde
    bundle = {'models': models, 'scaler': scaler}
    short = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
    with open(config.MODELS_DIR / f"{short}_bundle.pkl", 'wb') as f:
        pickle.dump(bundle, f)
    np.save(config.MODELS_DIR / f"{short}_val_preds.npy", val_preds)
    
    return auc



# Entraînement des 5 modèles 

results = {}

# 1. extraTrees
try:
    results['ExtraTrees'] = train_br(
        lambda: ExtraTreesClassifier(n_estimators=100, max_depth=14,
                                     min_samples_leaf=5, class_weight='balanced',
                                     n_jobs=6, random_state=config.RANDOM_STATE),
        X_tr, Y_tr, X_val, "ExtraTrees", needs_scaling=False)
except Exception as e:
    print(f"ExtraTrees ÉCHEC GLOBAL : {e}\n")

# 2. HistGradientBoosting
try:
    results['HistGB'] = train_br(
        lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                               max_depth=8, random_state=config.RANDOM_STATE),
        X_tr, Y_tr, X_val, "HistGB", needs_scaling=False)
except Exception as e:
    print(f"HistGB ÉCHEC GLOBAL : {e}\n")

# 3. Logistic Regression
try:
    results['LogReg'] = train_br(
        lambda: LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=1000, solver='lbfgs', n_jobs=6),
        X_tr, Y_tr, X_val, "LogReg", needs_scaling=True)
except Exception as e:
    print(f"LogReg ÉCHEC GLOBAL : {e}\n")

# 4. Gaussian Naive Bayes
try:
    results['GaussianNB'] = train_br(
        lambda: GaussianNB(),
        X_tr, Y_tr, X_val, "GaussianNB", needs_scaling=True)
except Exception as e:
    print(f"GaussianNB ÉCHEC GLOBAL : {e}\n")

# 5. Nyström + SGD (approximation de noyau RBF = méthode à noyau)
try:
    results['NystroemSVM'] = train_br(
        lambda: make_pipeline(
            Nystroem(kernel='rbf', gamma=1e-3, n_components=300,
                     random_state=config.RANDOM_STATE),
            SGDClassifier(loss='log_loss', class_weight='balanced',
                          max_iter=1000, random_state=config.RANDOM_STATE)
        ),
        X_tr, Y_tr, X_val, "NystroemSVM", needs_scaling=True)
except Exception as e:
    print(f"NystroemSVM ÉCHEC GLOBAL : {e}\n")


print("=" * 50)
print("RÉCAPITULATIF DES MODÈLES SUPPLÉMENTAIRES")
print("=" * 50)
print(f"  {'LGB (référence)':<20s} : 0.8591")
print(f"  {'RF  (référence)':<20s} : 0.8301")
for name, auc in results.items():
    print(f"  {name:<20s} : {auc:.4f}")