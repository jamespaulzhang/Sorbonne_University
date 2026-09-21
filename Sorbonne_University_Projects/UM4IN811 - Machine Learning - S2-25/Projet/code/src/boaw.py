"""
BoAW avec Bagging
pour echantillonner des trames mel à partir de toutes les pistes d'entraînement
et apprendre plusieurs codebooks K-means 
Pour chaque clip, calculer un histogramme de codes par codebook, puis concaténer les histogrammes.

"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import config


class BaggingBoAW:

    def __init__(self, n_codebooks: int = config.N_CODEBOOKS,
                 k: int = config.BOAW_K,
                 random_state: int = config.RANDOM_STATE):
        self.n_codebooks = n_codebooks
        self.k = k
        self.random_state = random_state
        self.codebooks = []  # liste de (kmeans, scaler)
    
    @property
    def feature_dim(self) -> int:
        #Dimension du vecteur BoAW final (= n_codebooks * k)
        return self.n_codebooks * self.k
    
    def fit(self, training_frames: np.ndarray) -> 'BaggingBoAW':

       # training_frames : matrice (n_total_frames, n_mels) regroupant des trames échantillonnées dans tous les clips d'entraînement. 
       # donc chaque ligne est une trame mel.
    
        n = len(training_frames)
        rng = np.random.RandomState(self.random_state)
        
        print(f"Apprentissage de {self.n_codebooks} codebooks "
              f"(k={self.k}) sur {n} trames...")
        
        for b in range(self.n_codebooks):

            idx = rng.choice(n, size=n, replace=True)
            frames_b = training_frames[idx]
            
            # Standardisation 
            sc = StandardScaler().fit(frames_b)
            frames_scaled = sc.transform(frames_b)
            
            # minibanth

            km = KMeans(
                n_clusters=self.k,
                random_state=self.random_state + b * 7,
                n_init=3,
                max_iter=150,
                verbose=0,
            )
            km.fit(frames_scaled)
            
            self.codebooks.append((km, sc))
            print(f"  Codebook {b+1}/{self.n_codebooks} terminé "
                  f"(inertia={km.inertia_:.0f})")
        
        return self
    
    def transform(self, log_mel: np.ndarray) -> np.ndarray:

        #Calcule le vecteur BoAW à partir de son spectrogramme log-mel.
        

        frames = np.nan_to_num(log_mel.T.astype(np.float32))
        hists = []
        for km, sc in self.codebooks:
            labels = km.predict(sc.transform(frames))
            # histogramme normalisé 
            h, _ = np.histogram(labels, bins=self.k, range=(0, self.k))
            h = h.astype(np.float32) / (h.sum() + 1e-8)
            hists.append(h)
        return np.concatenate(hists)


def sample_frames_for_codebook(audio_paths, n_frames_per_clip: int = config.BOAW_FRAMES_PER_CLIP,
                                verbose: bool = True) -> np.ndarray:

    #Échantillonne aléatoirement des trames log-mel à partir d'une liste de fichiers audio. 
    #pour constituer le jeu d'apprentissage des codebooks BoAW 


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
            t = log_mel.shape[1]
            sel = rng.choice(t, size=min(n_frames_per_clip, t), replace=False)
            all_frames.append(log_mel[:, sel].T)
        except Exception as e:
            if verbose:
                print(f"  Échec sur {path.name}: {e}")
            continue
        
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Échantillonnage : {i+1}/{n_files}")
    
    return np.nan_to_num(np.vstack(all_frames).astype(np.float32))