# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import matplotlib.pyplot as plt
import numpy as np

def modele_lin_analytique(X, y):
    """
    Calcule les paramètres d'un modèle linéaire (pente et ordonnée à l'origine)
    en utilisant les formules analytiques basées sur les statistiques des données.
    
    Paramètres:
    -----------
    X : array-like, vecteur des variables explicatives (abscisses)
    y : array-like, vecteur des variables à expliquer (ordonnées)
    
    Retour:
    -------
    ahat : float, estimation de la pente (coefficient a)
    bhat : float, estimation de l'ordonnée à l'origine (coefficient b)
    
    Formules:
    ---------
    â = cov(X,Y) / σ_x²
    b̂ = E(Y) - â * E(X)
    
    où:
    - cov(X,Y) est la covariance empirique non corrigée entre X et Y
    - σ_x² est la variance empirique non corrigée de X
    - E(X) et E(Y) sont les moyennes empiriques de X et Y respectivement
    
    Note:
    -----
    La covariance non corrigée correspond à la formule:
    cov(X,Y) = (1/n) * Σ (X_i - E(X)) * (Y_i - E(Y))
    (avec division par n et non par n-1)
    """
    # Calcul de la covariance empirique non corrigée
    # Utilisation de np.cov avec bias=True pour la version non corrigée
    covariance = np.cov(X, y, bias=True)[0, 1]  # [0,1] pour l'élément hors-diagonal
    
    # Calcul de la variance empirique non corrigée de X
    variance_x = np.var(X, ddof=0)  # ddof=0 pour la version non corrigée
    
    # Calcul des moyennes empiriques
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    
    # Application des formules analytiques
    ahat = covariance / variance_x
    bhat = mean_y - ahat * mean_x
    
    return ahat, bhat

def calcul_prediction_lin(X, a, b):
    """
    Calcule les valeurs prédites par un modèle linéaire.
    
    Paramètres:
    -----------
    X : array-like, vecteur des variables explicatives (abscisses)
    a : float, pente du modèle linéaire
    b : float, ordonnée à l'origine du modèle linéaire
    
    Retour:
    -------
    y_pred : array-like, vecteur des valeurs prédites
    
    Formule:
    --------
    y_pred = a * X + b
    """
    # Application de la formule linéaire y = a*x + b
    y_pred = a * X + b
    
    return y_pred

def erreur_mc(y_true, y_pred):
    """
    Calcule l'erreur au sens des moindres carrés (Mean Squared Error).
    
    Paramètres:
    -----------
    y_true : array-like, vecteur des valeurs réelles (observées)
    y_pred : array-like, vecteur des valeurs prédites
    
    Retour:
    -------
    mse : float, erreur quadratique moyenne
    
    Formule:
    --------
    MSE = (1/n) * Σ (y_true_i - y_pred_i)²
    
    où n est le nombre d'échantillons
    """
    # Calcul des différences entre valeurs réelles et prédites
    diff = y_true - y_pred
    
    # Calcul de la somme des carrés des différences
    somme_carres = np.sum(diff**2)
    
    # Normalisation par le nombre d'échantillons
    n = len(y_true)
    mse = somme_carres / n
    
    return mse

def dessine_reg_lin(X_train, y_train, X_test, y_test, a, b):
    """
    Dessine les nuages de points d'apprentissage et de test,
    ainsi que la droite de régression.
    Les points d'apprentissage sont connectés par une ligne continue.
    
    Paramètres:
    -----------
    X_train : array-like, abscisses de l'ensemble d'apprentissage
    y_train : array-like, ordonnées de l'ensemble d'apprentissage
    X_test : array-like, abscisses de l'ensemble de test
    y_test : array-like, ordonnées de l'ensemble de test
    a : float, pente de la droite de régression
    b : float, ordonnée à l'origine de la droite de régression
    
    Retour:
    -------
    None (affiche le graphique)
    """
    # Créer une figure et des axes
    plt.figure(figsize=(12, 8))
    
    # Tracer les points d'apprentissage
    plt.scatter(X_train, y_train, color='blue', alpha=0.7, 
                label='Train (points)', s=50, edgecolor='black', linewidth=0.5, zorder=10)
    
    # Connecter les points d'apprentissage avec une ligne continue
    # Pour éviter les lignes croisées, on trie d'abord les points selon X
    sorted_indices = np.argsort(X_train)
    X_train_sorted = X_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]
    plt.plot(X_train_sorted, y_train_sorted, 'b-', alpha=0.7, linewidth=1.5, 
             label='Train (connecté)', zorder=8)
    
    # Tracer les points de test
    plt.scatter(X_test, y_test, color='red', alpha=0.3, 
                label='Test', s=30, edgecolor='black', linewidth=0.5, zorder=5)
    
    # Calculer et tracer la droite de régression
    # Créer des points pour la droite (couverture de tout l'intervalle des x)
    x_min = min(np.min(X_train), np.min(X_test))
    x_max = max(np.max(X_train), np.max(X_test))
    
    # Ajouter une marge pour une meilleure visualisation
    margin = 0.05 * (x_max - x_min)
    x_line = np.linspace(x_min - margin, x_max + margin, 100)
    y_line = a * x_line + b
    
    plt.plot(x_line, y_line, color='green', linewidth=3, 
             label='Droite de régression', zorder=6, linestyle='-')
    
    # Ajouter l'équation de la droite
    eq_text = f'y = {a:.4f}·x + {b:+.4f}'
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), zorder=15)
    
    # Ajouter des informations supplémentaires
    info_text = f'N(train) = {len(X_train)}, N(test) = {len(X_test)}'
    plt.text(0.05, 0.88, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5), zorder=15)
    
    # Ajouter des labels et une légende
    plt.xlabel('Variable X', fontsize=14)
    plt.ylabel('Variable y', fontsize=14)
    plt.title('Régression Linéaire - Données et Droite de Régression', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Ajuster les limites des axes
    plt.xlim(x_min - margin, x_max + margin)
    
    # Afficher le graphique
    plt.tight_layout()
    plt.show()

def make_mat_lin_biais(X):
    """
    Crée la matrice enrichie (augmentée) Xe pour la régression linéaire.
    
    Paramètres:
    -----------
    X : array-like, vecteur des variables explicatives de forme (N,)
        où N est le nombre d'échantillons
    
    Retour:
    -------
    Xe : ndarray, matrice enrichie de forme (N, 2)
         La première colonne contient les valeurs originales de X
         La deuxième colonne contient des 1 (pour le terme constant/biais)
    
    Exemple:
    --------
    >>> X = np.array([0.1, 0.2, 0.3])
    >>> make_mat_lin_biais(X)
    array([[0.1, 1. ],
           [0.2, 1. ],
           [0.3, 1. ]])
    
    Formule:
    --------
    Xe = [X 1] où 1 est un vecteur colonne de 1 de même longueur que X
    """
    # Convertir X en array numpy si ce n'est pas déjà le cas
    X = np.asarray(X)
    
    # Obtenir le nombre d'échantillons
    N = len(X)
    
    # Créer la matrice enrichie Xe
    # Méthode 1: Utiliser column_stack
    Xe = np.column_stack([X, np.ones(N)])
    
    # Méthode alternative: Créer manuellement
    # Xe = np.zeros((N, 2))
    # Xe[:, 0] = X
    # Xe[:, 1] = 1.0
    
    return Xe

def reglin_matriciel(Xe, y):
    """
    Résout la régression linéaire par la méthode des moindres carrés en forme matricielle.
    
    Paramètres:
    -----------
    Xe : ndarray, matrice enrichie de forme (N, 2)
         La première colonne contient les valeurs de X
         La deuxième colonne contient des 1 (pour le terme constant/biais)
    y : array-like, vecteur des variables à expliquer de forme (N,)
    
    Retour:
    -------
    w : ndarray, vecteur des paramètres [a, b]
        où a est la pente et b l'ordonnée à l'origine
    
    Algorithme:
    -----------
    1. Calculer A = Xe^T * Xe
    2. Calculer B = Xe^T * y
    3. Résoudre A * w = B pour obtenir w
    
    Note:
    -----
    Cette méthode utilise la forme normale des équations des moindres carrés.
    """
    # Calculer A = Xe^T * Xe
    A = Xe.T @ Xe  # Produit matriciel
    
    # Calculer B = Xe^T * y
    B = Xe.T @ y
    
    # Résoudre le système linéaire A * w = B
    # np.linalg.solve utilise une méthode numérique stable
    w = np.linalg.solve(A, B)
    
    return w

def calcul_prediction_matriciel(Xe, w):
    """
    Calcule les valeurs prédites à partir de la matrice enrichie et des coefficients.
    
    Paramètres:
    -----------
    Xe : ndarray, matrice enrichie de forme (N, 2)
         La première colonne contient les valeurs de X
         La deuxième colonne contient des 1
    w : ndarray, vecteur des paramètres [a, b]
    
    Retour:
    -------
    y_pred : ndarray, vecteur des valeurs prédites de forme (N,)
    
    Formule:
    --------
    y_pred = Xe * w = a * X + b
    
    où X est la première colonne de Xe
    """
    # Calculer les prédictions par produit matriciel
    y_pred = Xe @ w
    
    return y_pred

def gen_data_poly2(a, b, c, sig, N, Ntest):
    """
    Génère des données polynomiales du second degré avec bruit gaussien.
    
    Paramètres:
    -----------
    a : float, coefficient du terme quadratique (x²)
    b : float, coefficient du terme linéaire (x)
    c : float, terme constant
    sig : float, écart type du bruit gaussien
    N : int, nombre de points d'apprentissage
    Ntest : int, nombre de points de test
    
    Retour:
    -------
    X_train, y_train : array, données d'apprentissage
    X_test, y_test : array, données de test
    
    Formule:
    --------
    y = a*x² + b*x + c + ε, où ε ∼ N(0, sig²)
    Les valeurs x sont générées avec linspace() dans [0, 1]
    """
    np.random.seed(0)
    
    # Générer les abscisses (x) dans [0, 1] avec linspace (déjà triées)
    X_train = np.linspace(0, 1, N)
    X_test = np.linspace(0, 1, Ntest)
    
    # Générer le bruit gaussien
    noise_train = np.random.randn(N) * sig
    noise_test = np.random.randn(Ntest) * sig
    
    # Calculer les ordonnées (y) selon la formule polynomiale
    y_train = a * X_train**2 + b * X_train + c + noise_train
    y_test = a * X_test**2 + b * X_test + c + noise_test
    
    return X_train, y_train, X_test, y_test

def make_mat_poly_biais(X):
    """
    Crée la matrice enrichie pour la régression polynomiale du second degré.
    
    Paramètres:
    -----------
    X : array-like, vecteur des variables explicatives de forme (N,)
    
    Retour:
    -------
    Xe : ndarray, matrice enrichie de forme (N, 3)
         Colonnes: [x², x, 1] pour le modèle y = a*x² + b*x + c
    
    Note:
    -----
    L'ordre des colonnes est [x², x, 1] pour correspondre aux coefficients
    w = [a, b, c] où a est le coefficient du terme quadratique.
    """
    X = np.asarray(X)
    N = len(X)
    
    # Construire la matrice enrichie avec les colonnes dans l'ordre [x², x, 1]
    Xe = np.column_stack([
        X**2,      # Colonne pour le terme quadratique (a)
        X,         # Colonne pour le terme linéaire (b)
        np.ones(N) # Colonne pour le terme constant (c)
    ])
    
    return Xe

def calcul_prediction_matriciel(Xe, w):
    """
    Calcule les valeurs prédites à partir de la matrice enrichie et des coefficients.
    
    Paramètres:
    -----------
    Xe : ndarray, matrice enrichie de forme (N, 3)
         Colonnes: [x², x, 1] pour la régression polynomiale
    w : ndarray, vecteur des paramètres [a, b, c]
    
    Retour:
    -------
    y_pred : ndarray, vecteur des valeurs prédites de forme (N,)
    
    Formule:
    --------
    y_pred = Xe * w = a*x² + b*x + c
    """
    # Calculer les prédictions par produit matriciel
    y_pred = Xe @ w
    
    return y_pred

def dessine_poly_matriciel(X_train, y_train, X_test, y_test, w):
    """
    Dessine les données et la courbe polynomiale ajustée par méthode matricielle.
    Les points d'apprentissage sont connectés par une ligne continue.
    
    Paramètres:
    -----------
    X_train : array, abscisses d'apprentissage
    y_train : array, ordonnées d'apprentissage
    X_test : array, abscisses de test
    y_test : array, ordonnées de test
    w : array, coefficients [a, b, c] du polynôme
         où a est le coefficient de x², b de x, c constant
    
    Retour:
    -------
    None (affiche le graphique)
    """
    # Extraire les coefficients
    a, b, c = w
    
    plt.figure(figsize=(12, 8))
    
    # Tracer les points d'apprentissage
    plt.scatter(X_train, y_train, color='blue', alpha=0.7, 
                label='Train (points)', s=50, edgecolor='black', linewidth=0.5, zorder=10)
    
    # Tracer les points de test
    plt.scatter(X_test, y_test, color='red', alpha=0.3, 
                label='Test (points)', s=30, edgecolor='black', linewidth=0.5, zorder=5)
    
    # Connecter les points d'apprentissage avec une ligne continue
    # Pour éviter les lignes croisées, on trie d'abord les points selon X
    sorted_indices = np.argsort(X_train)
    X_train_sorted = X_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]
    plt.plot(X_train_sorted, y_train_sorted, 'b-', alpha=0.7, linewidth=1.5, 
             label='Train (connecté)', zorder=8)
    
    # Tracer la courbe polynomiale ajustée
    x_min = min(np.min(X_train), np.min(X_test))
    x_max = max(np.max(X_train), np.max(X_test))
    
    margin = 0.05 * (x_max - x_min)
    x_curve = np.linspace(x_min - margin, x_max + margin, 200)
    y_curve = a * x_curve**2 + b * x_curve + c
    
    plt.plot(x_curve, y_curve, color='green', linewidth=3, 
             label='Ajustement polynomial', zorder=6, linestyle='-')
    
    # Ajouter l'équation du polynôme
    eq_text = f'y = {a:.4f}·x² + {b:+.4f}·x {c:+.4f}'
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), zorder=15)
    
    # Ajouter des informations supplémentaires
    info_text = f'N(train) = {len(X_train)}, N(test) = {len(X_test)}'
    plt.text(0.05, 0.88, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5), zorder=15)
    
    # Configuration du graphique
    plt.xlabel('Variable X', fontsize=14)
    plt.ylabel('Variable y', fontsize=14)
    plt.title('Régression Polynomiale - Méthode Matricielle', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(x_min - margin, x_max + margin)
    
    plt.tight_layout()
    plt.show()

def descente_grad_mc(X, y, eps=1e-4, nIterations=100):
    """
    Descente de gradient pour la régression linéaire
    X : matrice des données (N, d)
    y : vecteur cible (N,)
    eps : taux d'apprentissage
    nIterations : nombre d'itérations
    Retourne : w, wall (historique)
    """
    N, d = X.shape
    
    # Initialisation des poids
    w = np.zeros(d)
    
    # Pour stocker l'historique sous forme de tableau numpy
    wall = np.zeros((nIterations + 1, d))  # +1 pour inclure l'initialisation
    wall[0, :] = w.copy()
    
    for t in range(1, nIterations + 1):
        # prédiction
        ypred = X @ w
        
        # gradient de C(w)
        grad = -2 * X.T @ (y - ypred)
        
        # mise à jour
        w = w - eps * grad
        
        # stockage
        wall[t, :] = w.copy()
    
    return w, wall

def application_reelle(X_train, y_train, X_test, y_test):
    """
    Applique la régression linéaire sur des données réelles (prédiction de consommation de voitures).
    
    Paramètres:
    -----------
    X_train : ndarray, matrice enrichie d'apprentissage
    y_train : ndarray, vecteur cible d'apprentissage
    X_test : ndarray, matrice enrichie de test
    y_test : ndarray, vecteur cible de test
    
    Retour:
    -------
    w : ndarray, vecteur des poids du modèle
    yhat_train : ndarray, prédictions sur l'ensemble d'apprentissage
    yhat_test : ndarray, prédictions sur l'ensemble de test
    """
    # 1. Calculer les poids par résolution analytique
    # Formule: w = (X_train^T X_train)^(-1) X_train^T y_train
    A = X_train.T @ X_train
    B = X_train.T @ y_train
    w = np.linalg.solve(A, B)
    
    # 2. Calculer les prédictions
    yhat_train = X_train @ w
    yhat_test = X_test @ w
    
    # 3. Calculer les erreurs MSE
    err_train = np.mean((y_train - yhat_train)**2)
    err_test = np.mean((y_test - yhat_test)**2)
    
    # 4. Afficher les résultats avec le format exact du professeur
    print(f"w={w}")
    print(f"Erreur moyenne au sens des moindres carrés (train): {err_train:.2f}")
    print(f"Erreur moyenne au sens des moindres carrés (test): {err_test:.2f}")
    
    return w, yhat_train, yhat_test

def normalisation(X_train, X_test):
    """
    Normalisation des données (centrage-réduction)
    sur chaque colonne des matrices d'entrée.
    
    Paramètres
    ----------
    X_train : ndarray (N_train, d+1)
        Matrice d'apprentissage AVEC biais en dernière colonne.
    X_test : ndarray (N_test, d+1)
        Matrice de test AVEC biais en dernière colonne.

    Retour
    ------
    Xn_train : ndarray normalisé
    Xn_test : ndarray normalisé

    Remarques
    ---------
    - Seules les colonnes d'entrée (toutes sauf la dernière colonne 1 du biais)
      doivent être normalisées.
    - Le biais (colonne constante 1) NE DOIT PAS être modifié.
    - La moyenne et l'écart-type sont calculés UNIQUEMENT sur X_train.
    """

    # Copie pour ne pas modifier les originaux
    Xn_train = X_train.copy().astype(float)
    Xn_test  = X_test.copy().astype(float)

    # Indices des colonnes à normaliser (toutes sauf le biais)
    d = X_train.shape[1] - 1

    # 1) calcul μ_j et σ_j SUR LE TRAIN SEULEMENT
    mu = X_train[:, :d].mean(axis=0)
    sigma = X_train[:, :d].std(axis=0)

    # éviter la division par 0 si une colonne est constante
    sigma[sigma == 0] = 1.0

    # 2) normalisation : (X - μ) / σ
    Xn_train[:, :d] = (X_train[:, :d] - mu) / sigma
    Xn_test[:, :d]  = (X_test[:, :d]  - mu) / sigma

    return Xn_train, Xn_test

"""
Réponses aux questions d'ouverture
"""

# -------------------------------------------------------------
# Sélection de caractéristiques
# -------------------------------------------------------------
# En éliminant les variables qui contribuent peu à la prédiction, on observe
# généralement une amélioration de la stabilité du modèle et parfois une
# réduction du sur-apprentissage. Les résultats montrent que certaines
# variables continues (comme displacement ou horsepower) apportent moins
# d'information que le poids ou l'année du modèle. En les supprimant, la
# performance peut rester identique, voire s'améliorer légèrement si ces
# variables introduisaient du bruit.


# -------------------------------------------------------------
# Feature engineering : interprétation des variables
# -------------------------------------------------------------
# L'analyse des caractéristiques révèle que le poids, l'année du modèle et
# le biais (constante) sont des déterminants majeurs de la consommation (mpg).
# Jusqu'ici, la variable "origin" n'était pas exploitée car difficile à coder.


# -------------------------------------------------------------
# Encodage de l'origine
# -------------------------------------------------------------
# La variable « origine » (USA / Europe / Japon) est extraite via :
# origine = data.values[:, -2]
# Elle doit être encodée au début pour préserver la séparation apprentissage/test.
# Un encodage one-hot est approprié :
# origine = {1,2,3} → vecteur binaire de dimension 3.


# -------------------------------------------------------------
# Encodage de l'année
# -------------------------------------------------------------
# L'année du modèle peut aussi être encodée en one-hot, mais cela créerait
# trop de dimensions. On préfère regrouper les années en 10 catégories, puis
# les encoder en vecteur binaire :
# X_j = x ∈ {1,...,K} ⇒ vecteur e_j dans {0,1}^K


# -------------------------------------------------------------
# Impact de la normalisation sur le gradient
# -------------------------------------------------------------
# Oui, la normalisation affecte fortement le comportement du gradient.
# Sans normalisation, les dimensions peuvent avoir des échelles très
# différentes, ce qui provoque une descente en "zigzag" et ralentit la
# convergence. Avec normalisation, toutes les dimensions ont la même échelle :
# le choix du pas (learning rate) est plus simple et la convergence est plus
# régulière vers l'optimum.


# -------------------------------------------------------------
# Gradient stochastique
# -------------------------------------------------------------
# Dans le gradient stochastique (SGD), on met à jour les paramètres après
# chaque exemple tiré aléatoirement. Cela introduit du bruit dans la
# descente, mais permet d'explorer plus rapidement l'espace des paramètres.
# Sur les exemples jouets, on observe que :
# • la convergence est plus rapide en nombre d'itérations,
# • mais plus bruitée autour de l'optimum.


# -------------------------------------------------------------
# Amélioration du gradient : rôle du moment
# -------------------------------------------------------------
# Comme expliqué dans le blog de S. Ruder, l'ajout d'un "momentum" permet :
# • d'accélérer la convergence dans les vallées étroites,
# • de lisser les oscillations liées au gradient stochastique.
# La comparaison sur les données jouets montre que SGD+momentum
# atteint l'optimum plus rapidement et avec moins d'oscillations.