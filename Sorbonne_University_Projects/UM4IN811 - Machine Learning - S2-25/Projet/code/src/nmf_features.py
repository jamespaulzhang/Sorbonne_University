"""
 on apprend un dictionnaire global W de motifs spectraux récurrents,
puis pour chaque clip on calcule les activations H et on les résume par
statistiques. Ces activations capturent quelles briques sonores caractéristiques sont présentes dans le clip.

"""
import numpy as np
from sklearn.decomposition import NMF

from . import config


def _prepare_log_mel_for_nmf(log_mel: np.ndarray) -> np.ndarray:

    #Transforme un log-mel (n_mels, n_frames) en matrice (n_frames, n_mels) non-négative normalisée, pour la NMF.
    
 
    frames = log_mel.T.astype(np.float32)
    frames = frames - frames.min()
    frames = np.nan_to_num(frames)
    row_max = frames.max(axis=1, keepdims=True) + 1e-8
    frames = frames / row_max
    frames = np.clip(frames, 0, None)  
    return frames


class NMFFeatureExtractor:
  
    
    def __init__(self, n_components: int = config.NMF_COMPONENTS,
                 random_state: int = config.RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.nmf = None
    
    @property
    def feature_dim(self) -> int:
        return 3 * self.n_components
    
    def fit(self, training_frames: np.ndarray) -> 'NMFFeatureExtractor':

        #apprend le dictionnaire NMF.

        print(f"Apprentissage NMF ({self.n_components} composantes "
              f"sur {len(training_frames)} trames)...")
        self.nmf = NMF(
            n_components=self.n_components,
            init='nndsvda',
            max_iter=500,
            tol=1e-4,
            random_state=self.random_state,
        )
        self.nmf.fit(training_frames)
        print(f"  Erreur de reconstruction : {self.nmf.reconstruction_err_:.2f}")
        return self
    
    def transform(self, log_mel: np.ndarray) -> np.ndarray:

        #Calcule les activations H pour un clip et renvoie un résumé statistique.
        

        frames = _prepare_log_mel_for_nmf(log_mel)
        H = self.nmf.transform(frames)
        # 3 statistiques par composante : moyenne, écart-type, max
        feat = np.concatenate([H.mean(axis=0), H.std(axis=0), H.max(axis=0)])
        return np.nan_to_num(feat.astype(np.float32))


def sample_frames_for_nmf(audio_paths, n_frames_per_clip: int = config.NMF_FRAMES_PER_CLIP,
                           verbose: bool = True) -> np.ndarray:

    #Échantillonne des trames préparées pour la NMF (déjà rendues non-négativeset normalisées).

    from . import audio as audio_mod
    from . import features as feat_mod
    
    all_frames = []
    n_files = len(audio_paths)
    rng = np.random.RandomState(config.RANDOM_STATE)
    
    for i, path in enumerate(audio_paths):
        try:
            y = audio_mod.load_full_audio(path)
            y = audio_mod.find_best_window(y)
            log_mel = feat_mod.compute_log_mel(y)
            prepared = _prepare_log_mel_for_nmf(log_mel)
            t = prepared.shape[0]
            sel = rng.choice(t, size=min(n_frames_per_clip, t), replace=False)
            all_frames.append(prepared[sel])
        except Exception as e:
            if verbose:
                print(f"  Échec sur {path.name}: {e}")
            continue
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Échantillonnage NMF : {i+1}/{n_files}")
    
    return np.nan_to_num(np.vstack(all_frames).astype(np.float32))