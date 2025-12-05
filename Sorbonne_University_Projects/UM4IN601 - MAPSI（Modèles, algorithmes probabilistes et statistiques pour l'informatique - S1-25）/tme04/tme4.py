# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
from math import pi
import os
import pickle as pkl

def normale_bidim(x, mu, Sig):
    """
    Calcule la densité de probabilité d'une distribution normale bidimensionnelle
    
    Args:
        x: Point de données (vecteur 2D)
        mu: Vecteur moyenne de la distribution (vecteur 2D)
        Sig: Matrice de covariance (matrice 2x2)
    
    Returns:
        float: Densité de probabilité au point x
    """
    N = 2  # Dimension des données
    diff = x - mu  # Différence entre le point et la moyenne
    det_Sig = np.linalg.det(Sig)  # Déterminant de la matrice de covariance
    inv_Sig = np.linalg.inv(Sig)  # Inverse de la matrice de covariance
    exponent = -0.5 * diff @ inv_Sig @ diff.T  # Terme exponentiel de la formule
    normalization = 1 / ((2 * pi) ** (N / 2) * np.sqrt(det_Sig))  # Constante de normalisation
    return normalization * np.exp(exponent)

"""
L'ordonnancement des valeurs était PARFAITEMENT PRÉVISIBLE.
        
1. PROPRIÉTÉ FONDAMENTALE DE LA DISTRIBUTION NORMALE:
    - La densité de probabilité est MAXIMALE au point μ
    - Elle décroît exponentiellement avec la distance à μ
        
2. ANALYSE GÉOMÉTRIQUE:
    - Point x₁ = [1,2] : distance à μ = 0 → position optimale
    - Point x₂ = [0,0] : distance à μ = √5 ≈ 2.236 → position décentrée
        
3. IMPLICATION MATHÉMATIQUE:
    - Dans la formule exponentielle: e^(-½(x-μ)ᵀΣ⁻¹(x-μ))
    - Pour x=μ: l'exposant est 0 → e⁰ = 1 (maximum)
    - Pour x≠μ: l'exposant est négatif → e^(négatif) < 1
"""

def estimation_nuage_haut_gauche():
    """
    Estime manuellement les paramètres pour le nuage de points en haut à gauche
    
    Returns:
        tuple: (mu4, Sig4) - Moyenne et matrice de covariance estimées
    """
    mu4 = np.array([4.25, 80])  # Estimation de la moyenne du nuage
    
    # Estimation des écarts-types
    std_x = 0.5   # Écart-type pour l'axe des x (durée d'éruption)
    std_y = 6.67  # Écart-type pour l'axe des y (intervalle entre éruptions)
    
    # Conversion des écarts-types en variances
    var_x = std_x ** 2
    var_y = std_y ** 2
    
    # Estimation de la covariance (nulle pour cette estimation)
    cov = 0
    
    # Construction de la matrice de covariance
    Sig4 = np.array([[var_x, cov],
                     [cov, var_y]])
    
    return mu4, Sig4

def init(X, n_classes=2):
    """
    Initialise les paramètres pour l'algorithme EM
    
    Args:
        X: Matrice des données (n_samples x 2)
        n_classes: Nombre de classes (默认为2)
    
    Returns:
        tuple: (pi, mu, Sig) - Paramètres initiaux
    """
    # Initialisation des probabilités a priori (équiprobabilité)
    pi = np.ones(n_classes) / n_classes
    
    # Calcul de la moyenne globale des données
    overall_mean = np.mean(X, axis=0)
    
    # Initialisation des moyennes pour chaque classe
    mu = np.zeros((n_classes, 2))
    
    if n_classes == 2:
        offsets = [[1.0, 1.0], [-1.0, -1.0]]
    elif n_classes == 4:
        offsets = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    else:
        angles = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
        offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * 2
    
    for k in range(n_classes):
        mu[k] = overall_mean + offsets[k]
    
    # Calcul de la covariance globale
    overall_cov = np.cov(X.T)
    
    # Initialisation des matrices de covariance pour chaque classe
    Sig = np.zeros((n_classes, 2, 2))
    for k in range(n_classes):
        Sig[k] = overall_cov
    
    return pi, mu, Sig

def Q_i(X, pi, mu, Sig):
    """
    Étape E de l'algorithme EM : calcul des probabilités a posteriori
    
    Args:
        X: Matrice des données (n_samples x 2)
        pi: Probabilités a priori des classes
        mu: Moyennes des classes
        Sig: Matrices de covariance des classes
    
    Returns:
        np.array: Matrice des probabilités a posteriori (n_classes x n_samples)
    """
    n_samples = X.shape[0]
    n_classes = len(pi)
    
    # Initialisation de la matrice des probabilités a posteriori
    q = np.zeros((n_classes, n_samples))
    
    # Pour chaque point de données
    for i in range(n_samples):
        x = X[i]  # Point de données courant
        
        # Calcul des numérateurs (probabilités non normalisées)
        numerators = np.zeros(n_classes)
        for k in range(n_classes):
            # Calcul de la vraisemblance pour la classe k
            likelihood = normale_bidim(x, mu[k], Sig[k])
            # Produit avec la probabilité a priori
            numerators[k] = likelihood * pi[k]
        
        # Calcul du dénominateur (constante de normalisation)
        denominator = np.sum(numerators)
        
        # Normalisation pour obtenir les probabilités a posteriori
        for k in range(n_classes):
            q[k, i] = numerators[k] / denominator
    
    return q

def update_param(X, q, pi, mu, Sig):
    """
    Étape M de l'algorithme EM : mise à jour des paramètres
    Version générique sans seuil de stabilité numérique
    
    Args:
        X: Matrice des données (n_samples x 2)
        q: Probabilités a posteriori de l'étape E (forme: (n_classes, n_samples))
        pi: Anciennes probabilités a priori (longueur: n_classes)
        mu: Anciennes moyennes (forme: (n_classes, 2))
        Sig: Anciennes matrices de covariance (forme: (n_classes, 2, 2))
    
    Returns:
        tuple: (pi_new, mu_new, Sig_new) - Paramètres mis à jour
    """
    n_samples = X.shape[0]
    n_classes = len(pi)
    
    # Initialisation des nouveaux paramètres
    pi_new = np.zeros(n_classes)
    mu_new = np.zeros((n_classes, 2))
    Sig_new = np.zeros((n_classes, 2, 2))
    
    # Mise à jour pour chaque classe
    for k in range(n_classes):
        # Somme des poids pour la classe k
        sum_weights = np.sum(q[k, :])
        
        # Mise à jour de la probabilité a priori
        pi_new[k] = sum_weights / n_samples
        
        # Mise à jour de la moyenne (moyenne pondérée)
        weighted_sum = np.sum(q[k, :].reshape(-1, 1) * X, axis=0)
        mu_new[k] = weighted_sum / sum_weights
        
        # Mise à jour de la matrice de covariance
        diff = X - mu_new[k]  # Différences par rapport à la nouvelle moyenne
        weighted_outer = np.zeros((2, 2))
        
        # Calcul de la somme pondérée des produits externes
        for i in range(n_samples):
            outer_prod = np.outer(diff[i], diff[i])  # Produit externe
            weighted_outer += q[k, i] * outer_prod  # Pondération par q
        
        Sig_new[k] = weighted_outer / sum_weights  # Normalisation
    
    return pi_new, mu_new, Sig_new

def EM(X, initFunc=init, nIterMax=100, saveParam=None):
    """
    Algorithme EM complet pour les mixtures gaussiennes avec sauvegarde des paramètres
    
    Args:
        X: Matrice des données
        initFunc: Fonction d'initialisation des paramètres
        nIterMax: Nombre maximum d'itérations
        saveParam: Chemin pour sauvegarder les paramètres à chaque itération
    
    Returns:
        tuple: (nIter, pi, mu, Sig) - Résultats de l'algorithme
    """
    # Initialisation des paramètres
    pi, mu, Sig = initFunc(X)
    
    # Sauvegarde des paramètres initiaux si demandé
    if saveParam is not None:
        # Création du sous-répertoire si nécessaire
        directory = saveParam[:saveParam.rfind('/')]
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Sauvegarde des paramètres initiaux (itération 0)
        pkl.dump({'pi': pi, 'mu': mu, 'Sig': Sig},
                 open(saveParam + "0.pkl", 'wb'))
    
    # Boucle principale de l'algorithme EM
    for nIter in range(1, nIterMax + 1):
        mu_old = mu.copy()  # Sauvegarde pour test de convergence
        
        # Étape E : calcul des probabilités a posteriori
        q = Q_i(X, pi, mu, Sig)
        
        # Étape M : mise à jour des paramètres
        pi, mu, Sig = update_param(X, q, pi, mu, Sig)
        
        # Sauvegarde des paramètres si demandé
        if saveParam is not None:
            pkl.dump({'pi': pi, 'mu': mu, 'Sig': Sig},
                     open(saveParam + str(nIter) + ".pkl", 'wb'))
        
        # Test de convergence basé sur le changement des moyennes
        mu_change = np.max(np.abs(mu - mu_old))
        if mu_change < 1e-3:  # Seuil de convergence
            break
    
    return nIter, pi, mu, Sig

def init_4(X):
    """
    Initialisation spécifique pour 4 classes
    """
    return init(X, n_classes=4)

def bad_init_4(X):
    """
    Initialisation dégradée pour 4 classes avec des décalages spécifiques
    
    Args:
        X: Matrice des données (n_samples x 2)
    
    Returns:
        tuple: (pi, mu, Sig) - Paramètres initiaux dégradés
    """
    n_classes = 4
    
    # Probabilités a priori équiprobables
    pi = np.ones(n_classes) / n_classes
    
    # Calcul de la moyenne globale des données
    overall_mean = np.mean(X, axis=0)
    
    # Initialisation des moyennes avec des décalages spécifiques et défavorables
    # Ces décalages sont choisis pour être éloignés des clusters naturels
    offsets = [
        [4.0, 2.0],   # Décalage important vers le haut-droite
        [3.0, 4.0],   # Décalage vers le haut
        [0.0, 0.0],   # Pas de décalage (reste au centre)
        [-5.0, 0.0]   # Décalage important vers la gauche
    ]
    
    # Initialisation des moyennes pour chaque classe
    mu = np.zeros((n_classes, 2))
    for k in range(n_classes):
        mu[k] = overall_mean + offsets[k]
    
    # Calcul de la covariance globale
    overall_cov = np.cov(X.T)
    
    # Initialisation des matrices de covariance pour chaque classe
    Sig = np.zeros((n_classes, 2, 2))
    for k in range(n_classes):
        Sig[k] = overall_cov.copy()
    
    print("Initialisation dégradée appliquée:")
    print(f"Décalages utilisés: {offsets}")
    print(f"Moyennes initiales:\n{mu}")
    
    return pi, mu, Sig

def logpobsBernoulli(X, theta):
    """
    Calcul du logarithme de la vraisemblance d'une image binaire sous un modèle Bernoulli.
    
    Args:
        X (np.ndarray): vecteur 1D binaire (longueur 256) représentant une image.
        theta (np.ndarray): vecteur 1D de probabilités (longueur 256), chaque élément dans [0,1].
    
    Retour:
        float: log p(X | theta)
    """
    eps = 1e-5
    # éviter log(0) en bornant theta
    theta_clipped = np.clip(theta, eps, 1 - eps)
    # somme des logs : x*log(theta) + (1-x)*log(1-theta)
    log_prob = np.sum(X * np.log(theta_clipped) + (1 - X) * np.log(1 - theta_clipped))
    return float(log_prob)

def init_B(X, n_classes=10, n_samples_per_class=3, seed=None):
    """
    Initialisation des paramètres d'un mélange de Bernoulli.
    
    Stratégie :
      - pi initial égalitaire (1 / n_classes)
      - pour chaque classe k, on moyenne n_samples_per_class images consécutives
        à partir du début; si on manque d'exemples on effectue un tirage aléatoire.
    
    Args:
        X (np.ndarray): matrice des données (n_samples, n_features)
        n_classes (int): nombre de classes (K)
        n_samples_per_class (int): nombre d'échantillons consécutifs pour initialiser chaque classe
        seed (int|None): graine pour la reproductibilité (si fournie)
    
    Retour:
        (pi, theta) :
          pi (np.ndarray): vecteur (n_classes,) des poids a priori
          theta (np.ndarray): matrice (n_classes, n_features) des paramètres Bernoulli
    """
    if seed is not None:
        np.random.seed(seed)
    n_samples, n_features = X.shape

    # priors égaux
    pi = np.ones(n_classes) / float(n_classes)
    theta = np.zeros((n_classes, n_features), dtype=float)

    for k in range(n_classes):
        start_idx = k * n_samples_per_class
        end_idx = start_idx + n_samples_per_class
        if end_idx <= n_samples:
            class_samples = X[start_idx:end_idx]
        else:
            # si on ne peut pas prendre suffisamment d'exemples consécutifs,
            # on tire aléatoirement (sans remise sauf si nécessaire)
            replace = n_samples < n_samples_per_class
            indices = np.random.choice(n_samples, n_samples_per_class, replace=replace)
            class_samples = X[indices]
        # moyenne simple (pas de lissage de Laplace ici)
        theta[k] = class_samples.mean(axis=0)

    # on retourne les paramètres initiaux (on ne force pas ici le clipping)
    return pi, theta

def Q_i_B(X, pi, theta):
    """
    E-step : calcul des postérieurs p(z=k | x_i) (soft assignments) en utilisant
    la technique numerique log-sum-exp pour la stabilité.
    
    Args:
        X (np.ndarray): matrice des données (n_samples, n_features)
        pi (np.ndarray): vecteur de priors (n_classes,)
        theta (np.ndarray): matrice des paramètres (n_classes, n_features)
    
    Retour:
        q (np.ndarray): matrice (n_classes, n_samples) où q[k, i] = p(z=k | x_i)
    """
    n_samples = X.shape[0]
    n_classes = theta.shape[0]

    # matrice des scores log (classes x échantillons)
    log_scores = np.zeros((n_classes, n_samples), dtype=float)

    for k in range(n_classes):
        t = theta[k]
        # bornage pour la stabilité numérique
        t = np.clip(t, 1e-5, 1 - 1e-5)
        # log-vraisemblance pour tous les échantillons (somme sur les features)
        log_lik = (X * np.log(t) + (1 - X) * np.log(1 - t)).sum(axis=1)  # shape (n_samples,)
        log_scores[k, :] = log_lik + np.log(max(pi[k], 1e-5))

    # normalisation par log-sum-exp pour chaque échantillon (colonne)
    s = np.max(log_scores, axis=0)                    # max_k log_scores[k, i]
    exp_shifted = np.exp(log_scores - s[None, :])    # on soustrait s pour la stabilité
    denom = exp_shifted.sum(axis=0)                  # somme des exponentielles par colonne
    q = exp_shifted / denom[None, :]                 # shape (n_classes, n_samples)

    return q

def update_param_B(X, q, pi, theta):
    """
    M-step : mise à jour des paramètres pi et theta à partir des postérieurs q.
    
    Args:
        X (np.ndarray): données (n_samples, n_features)
        q (np.ndarray): postérieurs (n_classes, n_samples)
        pi (np.ndarray): anciens priors (n_classes,) (non utilisé directement mais gardé pour compatibilité)
        theta (np.ndarray): anciens theta (n_classes, n_features) (utilisé si une classe n'a aucun point)
    
    Retour:
        (pi_new, theta_new)
    """
    n_samples = X.shape[0]
    n_classes = q.shape[0]
    n_features = X.shape[1]

    pi_new = np.zeros(n_classes, dtype=float)
    theta_new = np.zeros((n_classes, n_features), dtype=float)

    # Nk : comptages (soft) par classe
    Nk = q.sum(axis=1)  # shape (n_classes,)

    for k in range(n_classes):
        sum_weights = Nk[k]
        # nouveau prior = proportion d'exemples (soft) attribués à la classe k
        pi_new[k] = sum_weights / float(n_samples)
        if sum_weights > 0:
            # somme pondérée des features : somme_i q[k,i] * X[i, :]
            weighted_sum = (q[k, :][:, None] * X).sum(axis=0)  # shape (n_features,)
            theta_new[k] = weighted_sum / sum_weights
        else:
            # si la classe n'a reçu aucun poids, on conserve l'ancien theta
            theta_new[k] = theta[k]

    return pi_new, theta_new

def EM_B(X, initFunc=init_B, nIterMax=100, tol=1e-3, saveParam=None, verbose=False):
    """
    Algorithme EM pour un mélange de Bernoulli (avec assignments soft).
    
    Args:
        X (np.ndarray): données d'entraînement (n_samples, n_features)
        initFunc (callable): fonction d'initialisation initFunc(X) -> (pi, theta)
        nIterMax (int): nombre maximal d'itérations
        tol (float): critère d'arrêt sur la variation maximale de theta
        saveParam: chemin ou mécanisme pour sauvegarder les paramètres (optionnel)
        verbose (bool): si True, affiche des informations d'itération
    
    Retour:
        (nIter, pi, theta) : itérations réalisées et paramètres finaux
    """
    # initialisation
    pi, theta = initFunc(X)
    if verbose:
        print("init pi=", pi)
        print("init theta shape=", theta.shape)

    for nIter in range(1, nIterMax + 1):
        theta_old = theta.copy()

        # E-step (soft posteriors)
        q = Q_i_B(X, pi, theta)

        # M-step
        pi, theta = update_param_B(X, q, pi, theta)

        # critère de convergence : variation maximale des composantes de theta
        theta_change = np.max(np.abs(theta - theta_old))

        if verbose and (nIter % 10 == 0 or nIter <= 5):
            print(f"itération {nIter}: changement max de theta = {theta_change:.6f}")

        if theta_change < tol:
            if verbose:
                print(f"Convergence atteinte après {nIter} itérations.")
            return nIter, pi, theta

    if verbose:
        print(f"Nombre maximal d'itérations ({nIterMax}) atteint.")
    return nIterMax, pi, theta

def calcul_purete(X, Y, pi, theta):
    """
    Calcule la pureté (purity) de la partition induite par (pi, theta) sur les données X,
    en se servant des vraies étiquettes Y pour l'évaluation.

    Args:
        X (np.ndarray): données, shape (n_samples, n_features)
        Y (np.ndarray): vraies étiquettes entières, shape (n_samples,)
        pi (np.ndarray): vecteur des proportions a priori, shape (K,)
        theta (np.ndarray): paramètres Bernoulli par classe, shape (K, n_features)

    Retour:
        purete (np.ndarray): vecteur de puretés par cluster, shape (K,)
        poids  (np.ndarray): vecteur des tailles (poids) par cluster, shape (K,)
    """
    n_samples = X.shape[0]
    K = theta.shape[0]

    # --- 1) prédiction des labels \hat{Y} via affectation dure (argmax sur les log-scores) ---
    # calculer log-scores (K x n_samples)
    log_scores = np.zeros((K, n_samples), dtype=float)
    for k in range(K):
        t = np.clip(theta[k], 1e-5, 1 - 1e-5)
        # log-vraisemblance de chaque échantillon sous la classe k
        log_lik = (X * np.log(t) + (1 - X) * np.log(1 - t)).sum(axis=1)
        # ajouter log pi pour la composante a priori
        log_scores[k, :] = log_lik + np.log(max(pi[k], 1e-5))

    # affectation dure : pour chaque échantillon, on prend la classe argmax
    hat_assign = np.argmax(log_scores, axis=0)  # shape (n_samples,)

    # --- 2) pour chaque cluster c, récupérer les vraies étiquettes Y des points assignés ---
    purete = np.zeros(K, dtype=float)
    poids = np.zeros(K, dtype=int)

    for c in range(K):
        idx = np.where(hat_assign == c)[0]  # indices des échantillons dans le cluster c
        poids[c] = idx.shape[0]
        if poids[c] == 0:
            # cluster vide -> pureté = 0 (ou définir selon convention)
            purete[c] = 0.0
        else:
            Y_hat_c = Y[idx]
            # val: valeurs uniques, counts: leurs fréquences
            val, counts = np.unique(Y_hat_c, return_counts=True)
            # classe majoritaire et son nombre
            maj_count = counts.max()
            # pureté du cluster = pourcentage d'éléments qui appartiennent à la classe majoritaire
            purete[c] = float(maj_count) / float(poids[c])

    return purete, poids