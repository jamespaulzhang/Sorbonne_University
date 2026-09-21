"""
ici cest pour le chargement de fichiers .ogg et sélection de fenêtres de 5s.

on cherche la fenêtre de plus haute énergie RMS, ce qui correspond généralement à un cri.

"""
import numpy as np
import librosa
from pathlib import Path

from . import config


def load_full_audio(audio_path: Path) -> np.ndarray:
    #chargement
    y, _ = librosa.load(str(audio_path), sr=config.SR, mono=True)
    return y.astype(np.float32)


def find_best_window(y: np.ndarray, win_samples: int = config.N_SAMPLES) -> np.ndarray:

    n = len(y)
    
    # si audio plus court que la fenêtre 
    if n < win_samples:
        n_repeats = win_samples // n + 1
        y = np.tile(y, n_repeats)[:win_samples]
        return y.astype(np.float32)
    
    # si audio plus long 
    
    frame_len = 2048
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    
    # nb de frames correspondant à win_samples
    win_frames = max(1, win_samples // hop)
    
    #  la fenêtre couvre tout l'audio, on prend depuis le début
    if win_frames >= len(rms):
        return y[:win_samples].astype(np.float32)
    
    # Ssomme glissante de l'énergie RMS 
    cumsum = np.convolve(rms, np.ones(win_frames), mode='valid')
    best_frame_start = int(np.argmax(cumsum))
    

    best_sample_start = best_frame_start * hop
    best_sample_start = min(best_sample_start, n - win_samples)
    
    return y[best_sample_start:best_sample_start + win_samples].astype(np.float32)


def extract_window_at_offset(y_full: np.ndarray, offset_sec: float,
                              win_samples: int = config.N_SAMPLES) -> np.ndarray:
    
    #pour extrait une fenêtre de longueur fixe à partir d'un offset en secondes.
    
    # le test set Kaggle est découpé en segments de 5s identifiés par leur offset 
    # et Kaggle utilise des row_id du type 'fichier_5' pour désigner la fenêtre se terminant à 5s, donc l'offset de DÉBUT est offset_sec - 5.

    sr = config.SR
    start = int(offset_sec * sr)
    end = start + win_samples
    
    if end > len(y_full):
        # fenêtre dépasse la fin 
        start = max(0, len(y_full) - win_samples)
        end = start + win_samples
    
    y = y_full[start:end]
    
    #  si l'audio est plus court que la fenêtre
    if len(y) < win_samples:
        y = np.pad(y, (0, win_samples - len(y)))
    
    return y.astype(np.float32)