
#ajouter train_soundscapes à l'entraînement pour réduire le décalage de distribution (domain shift).


#Train = 35549 features iNaturalist (cache) + 50 fichiers soundscape
#Val   = 16 fichiers soundscape (jamais vus à l'entraînement)
     
#split 50/16 pour des fichiers soundscape 


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import pickle
import ast
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src import config
from src import feature_pipeline
from src import audio as audio_mod
from src import features as feat_mod


#chargement
data = np.load(config.CACHE_DIR / "features_full.npz")
X_inat = data['X']
valid_mask = data['valid_mask']

# modèles non-supervisés déjà entraînés
with open(config.MODELS_DIR / "boaw_model.pkl", 'rb') as f:
    boaw_model = pickle.load(f)
with open(config.MODELS_DIR / "nmf_model.pkl", 'rb') as f:
    nmf_model = pickle.load(f)

sub = pd.read_csv(config.DATA_DIR / "sample_submission.csv")
species_list = sub.columns[1:].tolist()
sp2idx = {sp: i for i, sp in enumerate(species_list)}
n_sp = len(species_list)

# multi-hot iNaturalist
train_df = pd.read_csv(config.DATA_DIR / "train.csv")
Y_inat_all = np.zeros((len(train_df), n_sp), dtype=np.float32)
for r_idx, row in enumerate(train_df.itertuples()):
    if row.primary_label in sp2idx:
        Y_inat_all[r_idx, sp2idx[row.primary_label]] = 1.0
    try:
        sec = ast.literal_eval(row.secondary_labels) if isinstance(row.secondary_labels, str) else []
        for s in sec:
            if s in sp2idx:
                Y_inat_all[r_idx, sp2idx[s]] = 1.0
    except (ValueError, SyntaxError):
        pass
Y_inat = Y_inat_all[valid_mask]
print(f"iNaturalist : X={X_inat.shape}, Y={Y_inat.shape}")



ts_dir = config.DATA_DIR / "train_soundscapes"
ts_labels = pd.read_csv(config.DATA_DIR / "train_soundscapes_labels.csv")

def end_to_sec(s):
    h, m, sec = s.split(':')
    return int(h)*3600 + int(m)*60 + int(sec)

# Vérité terrain : (file_stem, end_sec) -> multi-hot
truth = {}
for row in ts_labels.itertuples():
    stem = row.filename.replace('.ogg', '')
    end_sec = end_to_sec(row.end)
    vec = np.zeros(n_sp, dtype=np.float32)
    for sp in str(row.primary_label).split(';'):
        if sp in sp2idx:
            vec[sp2idx[sp]] = 1.0
    truth[(stem, end_sec)] = vec

labeled_files = sorted(ts_labels['filename'].unique())
print(f"\nFichiers soundscape étiquetés : {len(labeled_files)}")

def extract_window_features(y_window):
    hc_feat, log_mel = feat_mod.extract_handcrafted_features(y_window)
    boaw_feat = boaw_model.transform(log_mel)
    nmf_feat = nmf_model.transform(log_mel)
    return np.concatenate([hc_feat, boaw_feat, nmf_feat]).astype(np.float32)

print("Extraction des features soundscape...")
t0 = time.time()
X_ss, Y_ss, file_ss = [], [], []
for fi, fname in enumerate(labeled_files):
    apath = ts_dir / fname
    if not apath.exists():
        continue
    stem = fname.replace('.ogg', '')
    y_full = audio_mod.load_full_audio(apath)
    dur = len(y_full) / config.SR
    n_seg = max(1, int(np.ceil(dur / 5.0)))
    for seg in range(n_seg):
        end_sec = int((seg + 1) * 5.0)
        key = (stem, end_sec)
        if key not in truth:
            continue
        y_win = audio_mod.extract_window_at_offset(y_full, seg * 5.0)
        X_ss.append(np.nan_to_num(extract_window_features(y_win)))
        Y_ss.append(truth[key])
        file_ss.append(fname)
    if (fi + 1) % 10 == 0:
        print(f"  {fi+1}/{len(labeled_files)} ({time.time()-t0:.0f}s)")

X_ss = np.array(X_ss, dtype=np.float32)
Y_ss = np.array(Y_ss, dtype=np.float32)
file_ss = np.array(file_ss)
print(f"Soundscape : X={X_ss.shape}, Y={Y_ss.shape}")


rng = np.random.RandomState(42)
files_shuffled = rng.permutation(labeled_files)
n_val_files = max(1, len(files_shuffled) // 4)   
val_files = set(files_shuffled[:n_val_files])
train_ss_files = set(files_shuffled[n_val_files:])

ss_val_mask = np.array([f in val_files for f in file_ss])
ss_tr_mask = ~ss_val_mask

X_ss_tr, Y_ss_tr = X_ss[ss_tr_mask], Y_ss[ss_tr_mask]
X_ss_val, Y_ss_val = X_ss[ss_val_mask], Y_ss[ss_val_mask]
print(f"\nSoundscape train : {X_ss_tr.shape[0]} fenêtres ({len(train_ss_files)} fichiers)")
print(f"Soundscape val   : {X_ss_val.shape[0]} fenêtres ({len(val_files)} fichiers)")



# les deux jeux d'entraînement

SOUNDSCAPE_WEIGHT = 40   # chaque fenêtre soundscape compte comme 40

# : iNaturalist + soundscapes (pondérés via duplication)
X_combined = np.vstack([X_inat] + [X_ss_tr] * SOUNDSCAPE_WEIGHT)
Y_combined = np.vstack([Y_inat] + [Y_ss_tr] * SOUNDSCAPE_WEIGHT)
print(f"\nJeu combiné : X={X_combined.shape} "
      f"(dont {X_ss_tr.shape[0]*SOUNDSCAPE_WEIGHT} fenêtres soundscape pondérées)")


LGB_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
    'num_leaves': 31, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'min_child_samples': 20, 'verbose': -1,
    'n_jobs': 6, 'seed': 42,
}

def train_and_eval(X_tr, Y_tr, X_val, Y_val, label):
    print(f"\nEntraînement : {label} ")
    t0 = time.time()
    aucs = []
    n_trained = 0
    for j in range(n_sp):
        ytr = Y_tr[:, j]
        yval = Y_val[:, j]
        #  évaluer seulement les espèces présentes dans le val soundscape
        if yval.sum() < 1 or yval.sum() == len(yval):
            continue
        if ytr.sum() < 3:
            continue
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=ytr),
                            num_boost_round=100)
        pred = booster.predict(X_val)
        try:
            aucs.append(roc_auc_score(yval, pred))
            n_trained += 1
        except ValueError:
            pass
    print(f"  {label} : {(time.time()-t0)/60:.1f} min, {n_trained} espèces")
    print(f"  Macro AUC (val terrain) : {np.mean(aucs):.4f}")
    print(f"  Médiane                 : {np.median(aucs):.4f}")
    return np.mean(aucs), n_trained




# iNaturalist seul
auc_inat, n1 = train_and_eval(X_inat, Y_inat, X_ss_val, Y_ss_val,
                              "iNaturalist seul")

#iNaturalist + soundscapes 
auc_comb, n2 = train_and_eval(X_combined, Y_combined, X_ss_val, Y_ss_val,
                              "iNaturalist + soundscapes")

print("\n" + "="*55)
print("RÉSULTAT FINAL")
print("="*55)
print(f"  iNaturalist seul           : {auc_inat:.4f}")
print(f"  iNaturalist + soundscapes  : {auc_comb:.4f}")
print(f"  Gain                       : {auc_comb - auc_inat:+.4f}")
print(f"  (espèces évaluées : {n2})")