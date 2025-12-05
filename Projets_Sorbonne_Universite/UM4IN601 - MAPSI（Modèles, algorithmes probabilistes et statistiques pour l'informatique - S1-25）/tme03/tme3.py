# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
import matplotlib.pyplot as plt

def learnML_parameters(X_train, Y_train):
    """
    Apprend les paramètres (mu et sigma^2) du modèle gaussien par maximum de vraisemblance.
    
    Args:
        X_train (np.ndarray): données d'apprentissage, shape (N, 256)
        Y_train (np.ndarray): étiquettes correspondantes, shape (N,)

    Returns:
        mu (np.ndarray): moyennes des classes, shape (C, 256)
        sigma2 (np.ndarray): variances des classes, shape (C, 256)
    """
    num_classes = len(np.unique(Y_train))
    mu = np.zeros((num_classes, X_train.shape[1]))
    sigma2 = np.zeros((num_classes, X_train.shape[1]))

    for c in range(num_classes):
        X_c = X_train[Y_train == c]
        mu_c = np.mean(X_c, axis=0)
        sigma2_c = np.mean((X_c - mu_c)**2, axis=0)
        mu[c] = mu_c
        sigma2[c] = sigma2_c

    return mu, sigma2

def log_likelihood(x, mu, sig, defeps):
    """
    Calcule la log-vraisemblance d'une image pour une classe donnée.

    Args:
        x (np.ndarray): vecteur image, shape (D,)
        mu (np.ndarray): moyenne de la classe, shape (D,)
        sig (np.ndarray): variance de la classe, shape (D,)
        defeps (float): valeur minimale pour sigma (optionnel)

    Returns:
        float: log-vraisemblance de l'image pour la classe
    """
    if defeps > 0:
        sig = np.maximum(sig, defeps)
        res = -0.5 * np.sum(np.log(2 * np.pi * sig) + ((x - mu) ** 2) / sig)
    else:
        mask = sig > 0
        res = -0.5 * np.sum(np.log(2 * np.pi * sig[mask]) + ((x[mask] - mu[mask]) ** 2) / sig[mask])
    return res

def classify_image(x, mu, sig, defeps = -1):
    """
    Classifie une seule image en utilisant le modèle gaussien.

    Args:
        x (np.ndarray): vecteur image, shape (D,)
        mu (np.ndarray): moyennes des classes, shape (C, D)
        sig (np.ndarray): variances des classes, shape (C, D)
        defeps (float): valeur minimale pour sigma (optionnel)

    Returns:
        int: indice de la classe prédite
    """
    max_val = log_likelihood(x, mu[0], sig[0], defeps)
    flag = 0
    for i in range(10):
        ll = log_likelihood(x, mu[i], sig[i], defeps)
        if ll > max_val:
            max_val = ll
            flag = i
    return flag

def classify_all_images(x, mu, sig, defeps):
    """
    Classifie toutes les images d'un ensemble.

    Args:
        x (np.ndarray): données à classer, shape (N, D)
        mu (np.ndarray): moyennes des classes, shape (C, D)
        sig (np.ndarray): variances des classes, shape (C, D)
        defeps (float): valeur minimale pour sigma (optionnel)

    Returns:
        np.ndarray: tableau des classes prédites, shape (N,)
    """
    predictions = []
    for i in range(x.shape[0]):
        predictions.append(classify_image(x[i], mu, sig, defeps))
    return np.array(predictions)

def matrice_confusion(Y, Y_hat):
    """
    Calcule la matrice de confusion entre les étiquettes réelles et prédites.

    Args:
        Y (np.ndarray): étiquettes réelles, shape (N,)
        Y_hat (np.ndarray): étiquettes prédites, shape (N,)

    Returns:
        np.ndarray: matrice de confusion, shape (C, C)
    """
    C = len(np.unique(Y))
    M = np.zeros((C, C), dtype=int)

    for i in range(len(Y)):
        M[Y[i], Y_hat[i]] += 1

    return M

def classificationRate(Y, Y_hat):
    """
    Calcule le taux de bonne classification.

    Args:
        Y (np.ndarray): étiquettes réelles, shape (N,)
        Y_hat (np.ndarray): étiquettes prédites, shape (N,)

    Returns:
        float: proportion de bonnes prédictions (0 <= taux <= 1)
    """
    correct = np.sum(Y == Y_hat)
    total = len(Y)
    return correct / total

def classifTest(X_test, Y_test, mu, sig, defeps=-1):
    """
    Classifie un ensemble de test et analyse les erreurs.

    Args:
        X_test (np.ndarray): images de test, shape (N, D)
        Y_test (np.ndarray): étiquettes réelles, shape (N,)
        mu (np.ndarray): moyennes des classes, shape (C, D)
        sig (np.ndarray): variances des classes, shape (C, D)
        defeps (float): valeur minimale pour sigma (optionnel)

    Returns:
        tuple: indices des images mal classées (comme np.where)
    """
    N = X_test.shape[0]
    Y_hat = np.empty(N, dtype=int)

    # 1. Classifier toutes les images de test
    print("1- Classification de toutes les images de test ...")
    for i in range(N):
        Y_hat[i] = classify_image(X_test[i], mu, sig, defeps)

    # 2. Calcul du taux de bonne classification
    taux = classificationRate(Y_test, Y_hat)
    print(f"2- Taux de bonne classification : {taux}")

    # 3. Affichage de la matrice de confusion
    M = matrice_confusion(Y_test, Y_hat)
    print("3- Matrice de confusion :")
    plt.figure(figsize=(3,3))
    plt.imshow(M)

    # 4. Retourner les indices des images mal classées
    mal_classees = np.where(Y_test != Y_hat)
    return mal_classees

def binarisation(X):
    """
    Binarisation des images pour la modélisation Bernoulli.

    Args:
        X (np.ndarray): tableau d'images de taille (N, D)

    Returns:
        np.ndarray: tableau binaire de même taille (valeurs {0, 1})
    """

    # Normaliser sur [0,1]
    X = X / 255.0

    # Application de la binarisation
    Xb = (X > 0).astype(int)

    return Xb

def learnBernoulli(Xb_train, Y_train):
    """
    Apprentissage des paramètres du modèle de Bernoulli pour chaque classe.

    Args:
        Xb_train (np.ndarray): données binarisées (N, D)
        Y_train (np.ndarray): labels réels (N,)

    Returns:
        np.ndarray: matrice theta (C, D) contenant les p_j^(c)
                    probabilité qu’un pixel soit éclairé pour chaque classe.
    """
    classes = np.unique(Y_train)
    C = len(classes)
    D = Xb_train.shape[1]
    theta = np.zeros((C, D))

    for c in classes:
        Xc = Xb_train[Y_train == c]     # Sélection des images de la classe c
        Nc = Xc.shape[0]                # Nombre d'images dans la classe c
        # p_j = moyenne des 1 sur tous les pixels de la classe
        theta[c, :] = np.sum(Xc, axis=0) / Nc

    return theta

def logpobsBernoulli(x, theta, epsilon=1e-4):
    """
    Calcule le log de vraisemblance d'une image x par rapport à toutes les classes.

    Args:
        x (np.ndarray): image binaire, shape (D,)
        theta (np.ndarray): paramètres du modèle Bernoulli, shape (C, D)
        epsilon (float): seuil pour éviter log(0)

    Returns:
        np.ndarray: log-vraisemblance de l'image pour chaque classe, shape (C,)
    """
    # Seuillage pour éviter log(0)
    theta = np.clip(theta, epsilon, 1 - epsilon)
    
    # x shape = (D,), theta shape = (C, D)
    # calculer log-likelihood pour chaque classe
    loglik = np.sum(x * np.log(theta) + (1 - x) * np.log(1 - theta), axis=1)
    
    return loglik

# Remarque / Réponse à la question :
# Ce résultat (par ex. array([ 95.2894, -913.869, ...])) n'est pas normal au sens de la
# modélisation Bernoulli standard.  Pour un modèle Bernoulli avec x_j ∈ {0,1} et
# p_j ∈ (0,1), chaque terme x_j*log(p_j) + (1-x_j)*log(1-p_j) est ≤ 0,
# donc la log-vraisemblance totale doit être ≤ 0 (en pratique fortement négative).
#
# Explication de la "valeur étonnante" :
# - Ici la première composante positive indique que la formule a été évaluée sur des
#   données qui ne respectent pas l'hypothèse x_j ∈ {0,1}. En effet, si x_j > 1
#   (par exemple valeurs de pixels en 0..255), alors (1 - x_j) devient négatif.
#   Comme log(1 - p_j) ≤ 0, le produit (1 - x_j) * log(1 - p_j) devient positif,
#   ce qui peut générer de grandes contributions positives et aboutir à un
#   log-likelihood global positif.
#
# Ce qui explique vos observations concrètes :
# - logpobsBernoulli(Xb_train[0], ...) avec Xb_train binaire donne des valeurs
#   négatives (attendues).
# - logpobsBernoulli(X_train[0], ...) avec X_train non binarisé (0..255) peut
#   produire des valeurs positives (comme +95) à cause de la présence de x_j > 1.
#
# Recommandations / corrections pratiques :
# 1) Toujours appeler logpobsBernoulli avec des images binarisées (Xb) : x_j ∈ {0,1}.
# 2) Ajouter des assertions dans la fonction pour prévenir l'usage incorrect :
#       assert set(np.unique(x)).issubset({0,1}), "x doit être binaire {0,1}"
#    ou, si l'on veut être plus permissif, normaliser et binariser à l'entrée.
# 3) Toujours clipper θ dans (ε, 1-ε) avant le log (déjà fait ici) pour éviter log(0).
# 4) Si on veut accepter des images en niveaux de gris, expliciter la conversion :
#       x = (x / 255.0 > threshold).astype(int)
#    plutôt que d'appliquer directement la formule Bernoulli.
#
# En résumé : la valeur positive n'est pas une propriété du modèle Bernoulli mais
# provient d'une violation de l'hypothèse x_j ∈ {0,1} (ou d'un mauvais pré-traitement).
# Corriger l'entrée (binarisation/normalisation) ou ajouter des vérifications éliminera
# cette anomalie.

def classifBernoulliTest(Xb_test, Y_test, theta, epsilon=1e-4):
    """
    Classifie un ensemble de test en utilisant le modèle Bernoulli
    et analyse les erreurs.

    Args:
        Xb_test (np.ndarray): images de test binarisées, shape (N, D)
        Y_test (np.ndarray): étiquettes réelles, shape (N,)
        theta (np.ndarray): paramètres Bernoulli (C, D)
        epsilon (float): seuil pour éviter log(0) (optionnel)

    Returns:
        tuple: indices des images mal classées (comme np.where)
    """
    N = Xb_test.shape[0]
    Y_hat = np.empty(N, dtype=int)

    # 1. Classification de toutes les images de test
    print("1- Classification de toutes les images de test ...")
    for i in range(N):
        # Calculer la log-vraisemblance pour chaque classe
        loglik = logpobsBernoulli(Xb_test[i], theta, epsilon=epsilon)
        # Choisir la classe avec la log-vraisemblance maximale
        Y_hat[i] = np.argmax(loglik)

    # 2. Calcul du taux de bonne classification
    taux = classificationRate(Y_test, Y_hat)
    print(f"2- Taux de bonne classification : {taux}")

    # 3. Affichage de la matrice de confusion
    M = matrice_confusion(Y_test, Y_hat)
    print("3- Matrice de confusion :")
    plt.figure(figsize=(3,3))
    plt.imshow(M)
    plt.title("Matrice de confusion (Bernoulli)")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.colorbar()
    plt.show()

    # 4. Retourner les indices des images mal classées
    mal_classees = np.where(Y_test != Y_hat)
    return mal_classees

def learnGeom(Xg_train, Y_train, seuil=1e-4):
    """
    Apprentissage des paramètres du modèle géométrique pour chaque classe.
    Version corrigée pour correspondre aux résultats attendus.
    """
    classes = np.unique(Y_train)
    C = len(classes)
    D = Xg_train.shape[1]
    theta = np.zeros((C, D))
    
    for c in classes:
        Xc = Xg_train[Y_train == c]
        Nc = Xc.shape[0]
        
        for j in range(D):
            # Maximum de vraisemblance: p = 1 / E[X]
            mean_val = np.mean(Xc[:, j])
            
            if mean_val <= seuil:
                theta[c, j] = 1.0 - seuil
            else:
                theta[c, j] = min(1.0 / mean_val, 1.0 - seuil)
    
    return theta

def logpobsGeom(x, theta, seuil=1e-4):
    """
    Calcule la log-vraisemblance d'un profil x pour toutes les classes.
    Version corrigée.
    """
    C = theta.shape[0]
    loglik = np.zeros(C)
    
    for c in range(C):
        p_vec = np.clip(theta[c], seuil, 1.0 - seuil)
        
        # Loi géométrique: P(X=k) = (1-p)^(k) * p  pour k = 0,1,2,...
        # Mais nos données commencent à 1, donc on utilise:
        # P(X=k) = (1-p)^(k-1) * p  pour k = 1,2,3,...
        log_p = np.log(p_vec)
        log_1_minus_p = np.log(1.0 - p_vec)
        
        # Correction: (x-1) car nos données commencent à 1
        loglik[c] = np.sum((x - 1) * log_1_minus_p + log_p)
    
    return loglik

def classifyGeom(x, theta):
    """
    Classifie un profil x en choisissant la classe avec log-vraisemblance maximale.

    Args:
        x (np.ndarray): profil d'une image, shape (16,)
        theta (np.ndarray): paramètres géométriques, shape (C,16)

    Returns:
        int: classe prédite
    """
    loglik = logpobsGeom(x, theta)
    return np.argmax(loglik)