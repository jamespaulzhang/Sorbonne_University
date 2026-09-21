
#Pipeline d'extraction de features pour un ensemble de fichiers audio.

#L'extraction est parallélisée via joblib pour exploiter les 6 cœurs. (navida 2060))
#Les modèles BoAW et NMF sont entraînés sur un sous-échantillon de trames, puis appliqués à tous les clips.


from pathlib import Path
import time
import numpy as np
from joblib import Parallel, delayed
import pickle

from . import config
from . import audio as audio_mod
from . import features as feat_mod
from . import boaw as boaw_mod
from . import nmf_features as nmf_mod



def _extract_one_clip(audio_path: Path,
                      boaw_model: boaw_mod.BaggingBoAW,
                      nmf_model: nmf_mod.NMFFeatureExtractor) -> np.ndarray | None:

    #pipeline  pour un clip : charge l'audio, extrait la meilleure fenêtre de 5s, calcule handcrafted + BoAW + NMF, renvoie un vecteur 1D.
    

    try:
        y_full = audio_mod.load_full_audio(audio_path)
        y = audio_mod.find_best_window(y_full)

        hc_feat, log_mel = feat_mod.extract_handcrafted_features(y)
        
        boaw_feat = boaw_model.transform(log_mel)
        nmf_feat = nmf_model.transform(log_mel)
        
        return np.concatenate([hc_feat, boaw_feat, nmf_feat]).astype(np.float32)
    except Exception as e:
        print(f"  [ERREUR] {audio_path.name}: {type(e).__name__}: {e}")
        return None



def fit_unsupervised_models(audio_paths: list[Path],
                            n_codebooks: int = config.N_CODEBOOKS,
                            boaw_k: int = config.BOAW_K,
                            nmf_components: int = config.NMF_COMPONENTS,
                            frames_per_clip: int = config.BOAW_FRAMES_PER_CLIP):

    #entraine models sur un échantillon de trames extraites des fichiers d'entraînement
    
    #On utilise le MÊME échantillonnage de trames pour BoAW et NMF afin
    #d'économiser le coût de chargement audio 
    #NMF requiert une normalisation supplémentaire.

    print(f"\n=== Apprentissage BoAW + NMF ===")
    print(f"Échantillonnage des trames depuis {len(audio_paths)} fichiers...")
    t0 = time.time()
    
    # BoAW 
    boaw_frames = boaw_mod.sample_frames_for_codebook(
        audio_paths, n_frames_per_clip=frames_per_clip, verbose=True
    )
    print(f"  Trames BoAW : {boaw_frames.shape} ({time.time()-t0:.1f}s)")
    
    # NMF 

    t0 = time.time()
    nmf_frames = nmf_mod.sample_frames_for_nmf(
        audio_paths, n_frames_per_clip=frames_per_clip, verbose=True
    )
    print(f"  Trames NMF : {nmf_frames.shape} ({time.time()-t0:.1f}s)")
    

    boaw_model = boaw_mod.BaggingBoAW(n_codebooks=n_codebooks, k=boaw_k)
    boaw_model.fit(boaw_frames)
    

    nmf_model = nmf_mod.NMFFeatureExtractor(n_components=nmf_components)
    nmf_model.fit(nmf_frames)
    
    return boaw_model, nmf_model


def extract_features_parallel(audio_paths: list[Path],
                              boaw_model: boaw_mod.BaggingBoAW,
                              nmf_model: nmf_mod.NMFFeatureExtractor,
                              n_jobs: int = config.N_JOBS,
                              verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:

    #extrait les features de tous les fichiers en parallèle.
    

    n = len(audio_paths)
    if verbose:
        print(f"\n Extraction parallèle sur {n} fichiers ({n_jobs} workers) ")
    
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10 if verbose else 0)(
        delayed(_extract_one_clip)(p, boaw_model, nmf_model)
        for p in audio_paths
    )
    
    elapsed = time.time() - t0
    
    #  pour  sépare les réussites des échecs
    valid_mask = np.array([r is not None for r in results])
    X = np.stack([r for r in results if r is not None])
    
    if verbose:
        print(f"\n  Terminé en {elapsed:.1f}s ({elapsed/n*1000:.1f} ms/clip)")
        print(f"  Réussites : {valid_mask.sum()} / {n}")
        print(f"  Échecs : {(~valid_mask).sum()}")
        print(f"  Matrice résultante : {X.shape}")
    
    return X, valid_mask



def cache_path(name: str) -> Path:
    """Chemin d'un fichier de cache .npz."""
    return config.CACHE_DIR / f"{name}.npz"


def save_features_cache(name: str, X: np.ndarray, valid_mask: np.ndarray,
                         boaw_model: boaw_mod.BaggingBoAW,
                         nmf_model: nmf_mod.NMFFeatureExtractor):
    #Sauvegarde des .pkl 

    npz_path = cache_path(name)
    np.savez_compressed(npz_path, X=X, valid_mask=valid_mask)
    print(f"Features sauvegardées : {npz_path}")
    print(f"  Taille : {npz_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    boaw_path = config.MODELS_DIR / "boaw_model.pkl"
    nmf_path = config.MODELS_DIR / "nmf_model.pkl"
    with open(boaw_path, 'wb') as f:
        pickle.dump(boaw_model, f)
    with open(nmf_path, 'wb') as f:
        pickle.dump(nmf_model, f)
    print(f"Modèles sauvegardés : {boaw_path.name}, {nmf_path.name}")


def load_features_cache(name: str):

    npz_path = cache_path(name)
    if not npz_path.exists():
        return None
    
    data = np.load(npz_path)
    X = data['X']
    valid_mask = data['valid_mask']
    
    with open(config.MODELS_DIR / "boaw_model.pkl", 'rb') as f:
        boaw_model = pickle.load(f)
    with open(config.MODELS_DIR / "nmf_model.pkl", 'rb') as f:
        nmf_model = pickle.load(f)
    
    print(f"Cache chargé : {npz_path.name} ({X.shape})")
    return X, valid_mask, boaw_model, nmf_model