
#Entraînement et inférence des trois familles de classifieurs : LightGBM (LGB)/Random Forest (RF) /LinearSVC + Platt 
#Approche Binary Relevance : un classifieur binaire par espèce



import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from . import config




LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbose': -1,
    'n_jobs': 6,  
    'seed': config.RANDOM_STATE,
}

LGB_N_ROUNDS = 200
LGB_EARLY_STOPPING = 30

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'n_jobs': 6,
    'random_state': config.RANDOM_STATE,
}

SVM_C = 0.5
SVM_MAX_ITER = 5000  # 2000 par défaut était insuffisant pour notre haute dimension
SVM_MIN_POSITIVES = 10  # seuil minimum d'exemples positifs pour entraîner un SVM


# LightGBM
def train_lgb_per_species(X_tr: np.ndarray, y_tr: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          species_list: list[str],
                          verbose: bool = True) -> tuple[dict, np.ndarray]:

    n_species = len(species_list)
    models = {}
    val_preds = np.zeros((len(X_val), n_species), dtype=np.float32)
    
    for i, sp in enumerate(species_list):
        y_tr_sp = y_tr[:, i]
        y_val_sp = y_val[:, i]

        n_pos_tr = int(y_tr_sp.sum())
        if n_pos_tr == 0 or n_pos_tr == len(y_tr_sp):
            models[sp] = None
            continue
        
        # Construction des datasets 
        train_set = lgb.Dataset(X_tr, label=y_tr_sp)
        
        # On utilise un valid set seulement s'il y a au moins 1 positif et 1 négatif
        callbacks = [lgb.log_evaluation(0)]
        if y_val_sp.sum() > 0 and y_val_sp.sum() < len(y_val_sp):
            valid_set = lgb.Dataset(X_val, label=y_val_sp, reference=train_set)
            callbacks.append(lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False))
            booster = lgb.train(
                LGB_PARAMS, train_set, num_boost_round=LGB_N_ROUNDS,
                valid_sets=[valid_set], callbacks=callbacks,
            )
        else:

            booster = lgb.train(
                LGB_PARAMS, train_set, num_boost_round=LGB_N_ROUNDS,
                callbacks=callbacks,
            )
        
        models[sp] = booster
        val_preds[:, i] = booster.predict(X_val)
        
        if verbose and (i + 1) % 30 == 0:
            print(f"  LGB : {i+1}/{n_species} espèces")
    
    return models, val_preds


def predict_lgb(models: dict, X: np.ndarray, species_list: list[str]) -> np.ndarray:
    """Prédit avec tous les modèles LGB. Espèces sans modèle → 0."""
    preds = np.zeros((len(X), len(species_list)), dtype=np.float32)
    for i, sp in enumerate(species_list):
        if models.get(sp) is not None:
            preds[:, i] = models[sp].predict(X)
    return preds



# Random Forest
def train_rf_per_species(X_tr: np.ndarray, y_tr: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         species_list: list[str],
                         verbose: bool = True) -> tuple[dict, np.ndarray]:

    n_species = len(species_list)
    models = {}
    val_preds = np.zeros((len(X_val), n_species), dtype=np.float32)
    
    for i, sp in enumerate(species_list):
        y_tr_sp = y_tr[:, i]
        n_pos_tr = int(y_tr_sp.sum())
        if n_pos_tr == 0 or n_pos_tr == len(y_tr_sp):
            models[sp] = None
            continue
        
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr_sp)
        models[sp] = rf
        val_preds[:, i] = rf.predict_proba(X_val)[:, 1]
        
        if verbose and (i + 1) % 30 == 0:
            print(f"  RF : {i+1}/{n_species} espèces")
    
    return models, val_preds


def predict_rf(models: dict, X: np.ndarray, species_list: list[str]) -> np.ndarray:
    """Prédit avec tous les modèles RF. Espèces sans modèle → 0."""
    preds = np.zeros((len(X), len(species_list)), dtype=np.float32)
    for i, sp in enumerate(species_list):
        if models.get(sp) is not None:
            preds[:, i] = models[sp].predict_proba(X)[:, 1]
    return preds



# LinearSVC 
class SVMBundle:

    
    def __init__(self):
        self.scaler = None
        self.models = {}
    
    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            species_list: list[str], verbose: bool = True) -> np.ndarray:

        n_species = len(species_list)
        val_preds = np.zeros((len(X_val), n_species), dtype=np.float32)
        
        # Scaler global, appris sur l'ensemble d'entraînement
        self.scaler = StandardScaler().fit(X_tr)
        X_tr_s = np.nan_to_num(self.scaler.transform(X_tr))
        X_val_s = np.nan_to_num(self.scaler.transform(X_val))
        
        for i, sp in enumerate(species_list):
            y_tr_sp = y_tr[:, i]
            n_pos_tr = int(y_tr_sp.sum())
            
            if n_pos_tr < SVM_MIN_POSITIVES or n_pos_tr == len(y_tr_sp):
                self.models[sp] = None
                continue
            
            base = LinearSVC(C=SVM_C, class_weight='balanced',
                 max_iter=SVM_MAX_ITER, random_state=config.RANDOM_STATE,
                 dual='auto')  
            # Calibration sigmoid (Platt) avec 3-fold CV interne
            cal = CalibratedClassifierCV(base, cv=min(3, n_pos_tr), method='sigmoid')
            cal.fit(X_tr_s, y_tr_sp)
            self.models[sp] = cal
            val_preds[:, i] = cal.predict_proba(X_val_s)[:, 1]
            
            if verbose and (i + 1) % 30 == 0:
                print(f"  SVM : {i+1}/{n_species} espèces "
                      f"(modèles entraînés : {sum(1 for m in self.models.values() if m is not None)})")
        
        return val_preds
    
    def predict(self, X: np.ndarray, species_list: list[str]) -> np.ndarray:
        """Prédit avec tous les modèles SVM. Espèces sans modèle → 0."""
        X_s = np.nan_to_num(self.scaler.transform(X))
        preds = np.zeros((len(X), len(species_list)), dtype=np.float32)
        for i, sp in enumerate(species_list):
            if self.models.get(sp) is not None:
                preds[:, i] = self.models[sp].predict_proba(X_s)[:, 1]
        return preds



# Ensemble : rank-averaging (plus robuste que weighted probability)


def rank_average(preds_list: list[np.ndarray]) -> np.ndarray:

    n_samples, n_species = preds_list[0].shape
    ranks_sum = np.zeros_like(preds_list[0], dtype=np.float32)
    
    for preds in preds_list:
        for j in range(n_species):
           
            ranks_sum[:, j] += preds[:, j].argsort().argsort() / max(1, n_samples - 1)
    
    return ranks_sum / len(preds_list)



# Évaluation


def macro_auc_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:

    from sklearn.metrics import roc_auc_score
    
    aucs = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0 and y_true[:, j].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
            except ValueError:
                continue
    
    return (float(np.mean(aucs)) if aucs else 0.0, len(aucs))