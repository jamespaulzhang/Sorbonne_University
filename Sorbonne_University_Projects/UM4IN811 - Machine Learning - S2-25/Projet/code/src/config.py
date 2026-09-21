import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # D:\birdclef_project
DATA_DIR = PROJECT_ROOT / "data" / "birdclef-2026"
AUDIO_TRAIN_DIR = DATA_DIR / "train_audio"
AUDIO_TEST_DIR = DATA_DIR / "test_soundscapes"

CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


for d in [CACHE_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# paramtre des audios

SR = 32000              
CLIP_SEC = 5.0          # 
N_SAMPLES = int(SR * CLIP_SEC)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 16000
N_MFCC = 20

N_CODEBOOKS = 3         # bagging  codebook 
BOAW_K = 256            # nb codebook cluster 
BOAW_FRAMES_PER_CLIP = 5  #  codebook chaque clip 


NMF_COMPONENTS = 64
NMF_FRAMES_PER_CLIP = 5


N_TRAIN_SAMPLES = None  
RANDOM_STATE = 42
VAL_RATIO = 0.20
N_JOBS = 6              # cpu nvidia 2060