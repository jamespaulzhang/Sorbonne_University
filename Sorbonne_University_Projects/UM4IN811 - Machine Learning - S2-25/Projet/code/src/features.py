
#extraction de descripteurs acoustiques traditionnels pour la classification multi-espèces.


import numpy as np
import librosa
from scipy.stats import skew, kurtosis

from . import config




def compute_log_mel(y: np.ndarray) -> np.ndarray:

    #Spectrogramme log-mel en dB, utlise par SegMel, MFCC, BoAW et NMF.


    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_stft_magnitude(y: np.ndarray) -> np.ndarray:

    #Magnitude STFT à fréquences LINÉAIRES. Forme : (n_fft/2+1, n_frames) = (1025, ~313).

    return np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))



def _stats4(matrix: np.ndarray) -> np.ndarray:

    return np.concatenate([
        matrix.mean(axis=1),
        matrix.std(axis=1),
        skew(matrix, axis=1),
        kurtosis(matrix, axis=1),
    ])


def _segment_stats(matrix: np.ndarray, n_segments: int = 3) -> np.ndarray:

    chunks = np.array_split(matrix, n_segments, axis=1)
    return np.concatenate([
        np.concatenate([c.mean(axis=1), c.std(axis=1)])
        for c in chunks
    ])



def seg_mel_features(log_mel: np.ndarray) -> np.ndarray:

    return _segment_stats(log_mel, n_segments=3)


def mfcc_features(log_mel: np.ndarray) -> np.ndarray:

    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.N_MFCC)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([_stats4(mfcc), _stats4(delta1), _stats4(delta2)])


def chroma_features(stft_mag: np.ndarray) -> np.ndarray:

    chroma = librosa.feature.chroma_stft(S=stft_mag, sr=config.SR)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)])


def spectral_contrast_features(stft_mag: np.ndarray) -> np.ndarray:

    #pour distinguer signaux harmoniques (forts pics)de signaux à large bande (bruit).


    contrast = librosa.feature.spectral_contrast(S=stft_mag, sr=config.SR, n_bands=6)
    return np.concatenate([contrast.mean(axis=1), contrast.std(axis=1)])


def rolloff_features(stft_mag: np.ndarray) -> np.ndarray:

    #fréquence en dessous de laquelle se trouve 85% de
    l'énergie. 

    ro = librosa.feature.spectral_rolloff(S=stft_mag, sr=config.SR)[0]
    return np.array([ro.mean(), ro.std(), skew(ro), kurtosis(ro)])


def bandwidth_features(stft_mag: np.ndarray) -> np.ndarray:

    # Bande passante spectrale : écart-type pondéré des fréquences par rapportau centroïde. 


    bw = librosa.feature.spectral_bandwidth(S=stft_mag, sr=config.SR)[0]
    return np.array([bw.mean(), bw.std(), skew(bw), kurtosis(bw)])


def rms_features(y: np.ndarray) -> np.ndarray:

    rms = librosa.feature.rms(y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)[0]
    return np.array([rms_db.mean(), rms_db.std(), skew(rms_db), kurtosis(rms_db)])


def zcr_features(y: np.ndarray) -> np.ndarray:

    #Taux de passages par zéro. Estimateur grossier de la fréquence dominanteet indicateur de bruit/signal voisé.


    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH
    )[0]
    return np.array([zcr.mean(), zcr.std(), skew(zcr), kurtosis(zcr)])


def envelope_features(y: np.ndarray) -> np.ndarray:

    #Caractérisation de l'enveloppe temporelle d'amplitude :


    rms = librosa.feature.rms(y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    rms = rms / (rms.max() + 1e-8)
    
    # Autocorrélation centrée et normalisée
    centered = rms - rms.mean()
    ac = np.correlate(centered, centered, mode='full')
    ac = ac[len(ac) // 2:][:40]
    ac = ac / (ac[0] + 1e-8)
    
    # Spectre de modulation (FFT de l'enveloppe)
    mod = np.abs(np.fft.rfft(rms, n=512))[:20]
    mod = mod / (mod.sum() + 1e-8)
    
    env_stats = np.array([rms.mean(), rms.std(), skew(rms), kurtosis(rms)])
    return np.concatenate([env_stats, ac, mod]).astype(np.float32)



FEATURE_GROUPS = [
    'seg_mel',       
    'mfcc',          
    'chroma',        
    'spec_contrast', 
    'rolloff',       
    'bandwidth',     
    'rms',           
    'zcr',           
    'envelope',      #
]



def extract_handcrafted_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    #extrait tous les descripteurs manuels d'une fenêtre audio de 5s.
    

    log_mel = compute_log_mel(y)
    stft_mag = compute_stft_magnitude(y)  
    
    feat = np.concatenate([
        seg_mel_features(log_mel),
        mfcc_features(log_mel),
        chroma_features(stft_mag),        
        spectral_contrast_features(stft_mag),  
        rolloff_features(stft_mag),        
        bandwidth_features(stft_mag),      
        rms_features(y),
        zcr_features(y),
        envelope_features(y),
    ])
    
    feat = np.nan_to_num(feat.astype(np.float32))
    return feat, log_mel