
#Entrée principale d'entraînement.


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import feature_pipeline
from src import models


def build_multihot_labels(train_df: pd.DataFrame, species_list: list[str]) -> np.ndarray:

    #Construit la matrice (n_clips, n_species) de labels multi-hot.

    import ast
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    n_species = len(species_list)
    Y = np.zeros((len(train_df), n_species), dtype=np.float32)
    
    for row_idx, row in enumerate(train_df.itertuples()):

        if row.primary_label in species_to_idx:
            Y[row_idx, species_to_idx[row.primary_label]] = 1.0

        try:
            sec = ast.literal_eval(row.secondary_labels) if isinstance(row.secondary_labels, str) else []
            for s in sec:
                if s in species_to_idx:
                    Y[row_idx, species_to_idx[s]] = 1.0
        except (ValueError, SyntaxError):
            pass
    
    return Y


def stratified_split(Y: np.ndarray, val_ratio: float = 0.20,
                     random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:

    #Split train/val "rare-aware" : on garantit qu'une espèce rare n'est pas intégralement dans le val set (sinon on ne peut pas l'apprendre).

    rng = np.random.RandomState(random_state)
    n = len(Y)

    primary = Y.argmax(axis=1)
    val_mask = np.zeros(n, dtype=bool)
    
    for sp_idx in range(Y.shape[1]):
        members = np.where(primary == sp_idx)[0]
        if len(members) <= 1:
            continue  # reste en train
        n_val = max(1, int(len(members) * val_ratio))
        n_val = min(n_val, len(members) - 1)  
        val_idx = rng.choice(members, size=n_val, replace=False)
        val_mask[val_idx] = True
    
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def main(debug: bool = False):
    t_start = time.time()
    
    print("=" * 60)
    print("BirdCLEF+ 2026 — entraînement")
    print("=" * 60)
    

    train_df = pd.read_csv(config.DATA_DIR / "train.csv")
    sub = pd.read_csv(config.DATA_DIR / "sample_submission.csv")
    species_list = sub.columns[1:].tolist()
    
    print(f"\nFichiers d'entraînement : {len(train_df)}")
    print(f"Espèces à prédire : {len(species_list)}")
    
    if debug:
        train_df = train_df.head(500).copy().reset_index(drop=True)
        print(f"\n*** MODE DEBUG : limité à {len(train_df)} fichiers ***")
    
    audio_paths = [config.AUDIO_TRAIN_DIR / f for f in train_df['filename']]
    

    # BoAW + NMF
  
    n_sample_for_codebook = min(3000, len(audio_paths))
    rng = np.random.RandomState(config.RANDOM_STATE)
    codebook_sample = [audio_paths[i] for i in rng.choice(len(audio_paths), n_sample_for_codebook, replace=False)]
    
    boaw_model, nmf_model = feature_pipeline.fit_unsupervised_models(
        codebook_sample,
        n_codebooks=config.N_CODEBOOKS,
        boaw_k=config.BOAW_K,
        nmf_components=config.NMF_COMPONENTS,
        frames_per_clip=config.BOAW_FRAMES_PER_CLIP,
    )
    

    #extraction des features

    cache_name = "features_debug" if debug else "features_full"
    cached = feature_pipeline.load_features_cache(cache_name)
    if cached is not None:
        X, valid_mask, boaw_model, nmf_model = cached
    else:
        X, valid_mask = feature_pipeline.extract_features_parallel(
            audio_paths, boaw_model, nmf_model, n_jobs=config.N_JOBS,
        )
        feature_pipeline.save_features_cache(cache_name, X, valid_mask, boaw_model, nmf_model)

    Y_all = build_multihot_labels(train_df, species_list)
    Y = Y_all[valid_mask]  # alignement avec X (qui exclut les fichiers en échec)
    print(f"\nMatrice X : {X.shape}, Y : {Y.shape}")
    print(f"Espèces avec ≥1 positif : {(Y.sum(axis=0) > 0).sum()} / {len(species_list)}")
    

    # Split train/val

    train_idx, val_idx = stratified_split(Y, val_ratio=config.VAL_RATIO)
    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]
    print(f"\nSplit : train={len(X_tr)}, val={len(X_val)}")
    
    # Entraînement des trois modèles

    print(f"\n LightGBM ")
    t0 = time.time()
    lgb_models, lgb_val_preds = models.train_lgb_per_species(
        X_tr, Y_tr, X_val, Y_val, species_list, verbose=True
    )
    auc_lgb, n_eval = models.macro_auc_skip_empty(Y_val, lgb_val_preds)
    print(f"LGB terminé en {(time.time()-t0)/60:.1f} min — macro AUC = {auc_lgb:.4f} ({n_eval} espèces évaluées)")
    
    print(f"\n=== Random Forest ===")
    t0 = time.time()
    rf_models, rf_val_preds = models.train_rf_per_species(
        X_tr, Y_tr, X_val, Y_val, species_list, verbose=True
    )
    auc_rf, _ = models.macro_auc_skip_empty(Y_val, rf_val_preds)
    print(f"RF terminé en {(time.time()-t0)/60:.1f} min — macro AUC = {auc_rf:.4f}")
    """
    print(f"\n=== Linear SVM + Platt ===")
    t0 = time.time()
    svm_bundle = models.SVMBundle()
    svm_val_preds = svm_bundle.fit(X_tr, Y_tr, X_val, Y_val, species_list, verbose=True)
    auc_svm, _ = models.macro_auc_skip_empty(Y_val, svm_val_preds)
    print(f"SVM terminé en {(time.time()-t0)/60:.1f} min — macro AUC = {auc_svm:.4f}")
    """
    print(f"\n=== SVM : DÉSACTIVÉ pour ce run ===")
    auc_svm = 0.0
    svm_val_preds = None
    svm_bundle = None

    # Filtrage des modèles trop faibles avant l'ensemble : un modèle proche
    # du hasard dégrade le rank-average au lieu de l'aider.

    candidates = [('LGB', lgb_val_preds, auc_lgb), ('RF', rf_val_preds, auc_rf)]
    preds_for_ensemble = [p for name, p, a in candidates if a >= 0.6]
    labels_for_ensemble = [name for name, p, a in candidates if a >= 0.6]

    if preds_for_ensemble:
        ens_preds = models.rank_average(preds_for_ensemble)
        auc_ens, _ = models.macro_auc_skip_empty(Y_val, ens_preds)
        print(f"\nEnsemble ({'+'.join(labels_for_ensemble)}) : {auc_ens:.4f}")
    else:
        auc_ens = 0.0

    suffix = "_debug" if debug else ""
    
    with open(config.MODELS_DIR / f"lgb_models{suffix}.pkl", 'wb') as f:
        pickle.dump(lgb_models, f)
    with open(config.MODELS_DIR / f"rf_models{suffix}.pkl", 'wb') as f:
        pickle.dump(rf_models, f)
    if svm_bundle is not None:
        with open(config.MODELS_DIR / f"svm_bundle{suffix}.pkl", 'wb') as f:
            pickle.dump(svm_bundle, f)
    with open(config.MODELS_DIR / f"species_list{suffix}.pkl", 'wb') as f:
        pickle.dump(species_list, f)
    
    print(f"Modèles sauvegardés dans {config.MODELS_DIR}")
    print(f"\nTemps total : {(time.time()-t_start)/60:.1f} min")
    # Sauvegarde des prédictions sur le val set + labels.
    # Permet de recalculer n'importe quelle métrique ou ensemble SANS réentraîner.
    np.save(config.MODELS_DIR / "Y_val.npy", Y_val)
    np.save(config.MODELS_DIR / "lgb_val_preds.npy", lgb_val_preds)
    np.save(config.MODELS_DIR / "rf_val_preds.npy", rf_val_preds)
    # On sauvegarde aussi les indices du split pour pouvoir aligner
    # d'autres modèles entraînés séparément sur exactement le même val set.
    np.save(config.MODELS_DIR / "val_idx.npy", val_idx)
    np.save(config.MODELS_DIR / "train_idx.npy", train_idx)
    print("Prédictions val + indices du split sauvegardés (pour évaluation ultérieure)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Mode mini : 500 fichiers')
    args = parser.parse_args()
    main(debug=args.debug)