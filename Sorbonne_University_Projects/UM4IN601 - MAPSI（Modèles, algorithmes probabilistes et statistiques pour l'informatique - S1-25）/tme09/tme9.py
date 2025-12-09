# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
import matplotlib.pyplot as plt

# Exercice 1
def labels_tobinary(Y, target_class):
    """
    Convertit des étiquettes multiclasses en étiquettes binaires
    
    Args:
    Y: tableau d'étiquettes original
    target_class: classe cible (classe positive)
    
    Returns:
    Tableau d'étiquettes binaires: classe cible = 1, autres = 0
    """
    return (Y == target_class).astype(int)

def pred_lr(X, w, b):
    """
    Fonction de prédiction pour la régression logistique
    
    Args
    X: matrice de caractéristiques (N, d)
    w: vecteur de poids (d,)
    b: biais scalaire
    
    Returns:
    Vecteur de probabilités P(y=1|x) de dimension (N,)
    """
    z = X.dot(w) + b
    return 1 / (1 + np.exp(-z))  # fonction sigmoïde

def classify_binary(probabilities):
    """
    Convertit des probabilités en étiquettes de classe binaires
    
    Args:
    probabilities: tableau de probabilités [P(y=1|x)]
    
    Returns:
    Tableau d'étiquettes binaires (0 ou 1)
    """
    return (probabilities > 0.5).astype(int)

def accuracy(y_pred, y_true):
    """
    Calcule le taux de bonne classification
    
    Args:
    y_pred: étiquettes prédites
    y_true: étiquettes vraies
    
    Returns:
    Taux de bonne classification (float entre 0 et 1)
    """
    return np.mean(y_pred == y_true)

def rl_gradient_ascent(X, Y, eta=1e-4, niter_max=300):
    """
    Entraîne un modèle de régression logistique par montée de gradient
    
    Args:
    X: matrice de caractéristiques (N, d)
    Y: vecteur d'étiquettes binaires (N,)
    eta: taux d'apprentissage
    niter_max: nombre maximal d'itérations
    
    Returns:
    w: vecteur de poids entraîné
    b: biais entraîné
    accs: tableau des taux de bonne classification à chaque itération
    it: nombre d'itérations effectuées
    """
    N, d = X.shape
    
    # Initialisation des paramètres
    w = np.zeros(d)  # initialisation à zéro
    b = 0
    
    # Historique des taux de bonne classification
    accs = []
    
    # Montée de gradient
    for it in range(niter_max):
        # Calcul de exp(-(Xw + b))
        z = X.dot(w) + b
        exp_term = np.exp(-z)
        
        # Calcul des probabilités P(y=1|x) = 1/(1+exp(-(Xw+b)))
        predictions = 1 / (1 + exp_term)
        
        # Calcul de l'erreur Y - predictions
        error = Y - predictions
        
        # Calcul du gradient (selon la formule)
        # ∇_w CE = X^T (Y - predictions)
        gradient_w = X.T.dot(error)
        
        # ∂/∂b CE = sum(Y - predictions)
        gradient_b = np.sum(error)
        
        # Mise à jour des paramètres (montée de gradient)
        w += eta * gradient_w
        b += eta * gradient_b
        
        # Calcul du taux de bonne classification courant
        binary_pred = (predictions > 0.5).astype(int)
        current_acc = np.mean(binary_pred == Y)
        accs.append(current_acc)

    return w, b, np.array(accs), niter_max

def visualization(w):
    """
    Visualise le vecteur de poids
    """
    # Reconstruction du vecteur 256D en image 16x16
    if len(w) == 256:
        img = w.reshape(16, 16)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.show()

# Exercice 2
def rl_gradient_ascent_one_against_all(X, Y, epsilon, niter_max):
    """
    Approche un-contre-tous pour classification multi-classe.
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels (N,) avec K classes
        epsilon: Pas d'apprentissage pour chaque classifieur binaire
        niter_max: Nombre maximum d'itérations
        
    Returns:
        W: Matrice de poids (d, K)
        B: Vecteur des biais (K,)
    """
    N, d = X.shape
    classes = np.unique(Y)
    K = len(classes)
    
    W = np.zeros((d, K))
    B = np.zeros(K)
    
    for k, cl in enumerate(classes):
        # Transformation des labels en binaire
        Y_bin = labels_tobinary(Y, cl)
        
        # Apprentissage du classifieur binaire
        w_k, b_k, _, _ = rl_gradient_ascent(X, Y_bin, eta=epsilon, niter_max=niter_max)
        
        # Stockage des paramètres
        W[:, k] = w_k
        B[k] = b_k
        
        # Calcul de l'accuracy pour cette classe
        Y_pred_prob = pred_lr(X, w_k, b_k)
        Y_pred = classify_binary(Y_pred_prob)
        acc = accuracy(Y_pred, Y_bin)
        print(f"Classe : {cl} acc train={acc*100:.2f} %")
    
    return W, B

def classif_multi_class(Y_pred_matrix):
    """
    Classification multi-classe par règle du maximum.
    
    Args:
        Y_pred_matrix: Matrice des probabilités (N, K)
        
    Returns:
        Labels prédits (N,)
    """
    return np.argmax(Y_pred_matrix, axis=1)

"""
Exercice 3 : Analyse qualitative des solutions

Question 1: Quels sont les pixels qui jouent un role dans la décision?

Réponse :
Les pixels qui jouent un rôle dans la décision sont ceux qui ont une valeur non nulle 
(positifs) dans l'image ET un poids correspondant non nul dans le modèle.

Plus précisément:
- Les pixels avec des valeurs élevées et des poids positifs contribuent positivement 
  à la décision pour une classe donnée.
- Les pixels avec des valeurs élevées et des poids négatifs contribuent négativement 
  (s'opposent à la décision pour cette classe).
- Les pixels noirs (valeurs proches de 0) n'ont aucune contribution, quelle que soit 
  la valeur du poids à cette position.

La visualisation par heatmap (carte de chaleur) montre précisément le produit terme à 
terme entre les valeurs des pixels et les poids du modèle: heatmap = X[i] * W[:, classe].
Cette visualisation permet d'identifier quels pixels soutiennent (valeurs positives) 
ou s'opposent (valeurs négatives) à la classification dans une classe spécifique.

Question 2: Quelle limitation sur l'encodage des pixels noirs (à 0) cette visualisation 
met-elle en évidence ? Expliquer.

Réponse :
La limitation principale révélée par cette visualisation est que les pixels noirs 
(valeurs égales à 0) masquent complètement l'information des poids correspondants 
dans le modèle.

Explication détaillée:
- Dans la heatmap, chaque élément est calculé comme: pixel_value * weight.
- Si pixel_value = 0, alors pixel_value * weight = 0, indépendamment de la valeur 
  du poids (même si weight est très grand, positif ou négatif).
- Cela signifie que les zones noires de l'image n'apparaissent jamais dans la heatmap, 
  même si le modèle a appris des poids importants à ces positions.
- Par conséquent, on ne peut pas savoir si un pixel noir correspond à un poids fort 
  qui serait important si le pixel devenait blanc, ou à un poids faible sans importance.
- Cette limitation peut conduire à une mauvaise interprétation des décisions du modèle, 
  en particulier pour les images avec beaucoup de pixels noirs (arrière-plan).

Implications:
1. Les poids importants dans les régions noires sont invisibles dans la heatmap.
2. La heatmap ne montre que l'interaction entre l'image spécifique et les poids, 
   pas l'importance générale des poids.
3. Pour analyser complètement le comportement du modèle, il faut compléter la heatmap 
   par une visualisation directe des poids (sans multiplication par les valeurs de pixels).

Cette limitation est particulièrement critique pour l'analyse des erreurs de 
classification, car elle peut masquer les vraies raisons des décisions du modèle.
"""

# Exercice 4
def normalize(X):
    """
    Normalise les données pixel de [0, 2] vers [-1, 1].
    
    Args:
        X: Données d'entrée (N, d)
        
    Returns:
        Xn: Données normalisées
    """
    # Les pixels sont entre 0 et 2, on soustrait 1 pour avoir [-1, 1]
    return X - 1

"""
Exercice 4 : Normalisation des données X

Explication de la normalisation X_n = X - 1 et comment elle résout le problème précédent.

Dans l'exercice précédent, nous avons identifié une limitation importante dans la visualisation par heatmap :
les pixels noirs (valeur = 0) masquent complètement l'information des poids correspondants dans le modèle.

PROBLÈME PRÉCÉDENT :
-------------------
1. Dans les données USPS originales, les valeurs des pixels sont comprises entre 0 et 2.
   - 0 : noir (arrière-plan)
   - 2 : blanc (chiffre)
   - 1 : niveaux de gris intermédiaires

2. La heatmap était calculée comme : heatmap = X * W
   - Si X[i] = 0 (pixel noir) → heatmap[i] = 0, quel que soit W[i]
   - Conséquence : Les poids importants dans les zones noires étaient invisibles

3. Cela conduisait à :
   - Une interprétation incomplète des décisions du modèle
   - L'impossibilité de voir l'importance des poids dans l'arrière-plan
   - Une analyse biaisée des erreurs de classification

COMMENT LA NORMALISATION X_n = X - 1 RÉSOUT CE PROBLÈME :
---------------------------------------------------------
En soustrayant 1 à toutes les valeurs de pixels :
1. L'intervalle des valeurs passe de [0, 2] à [-1, 1]
   - Ancienne valeur 0 (noir) → Nouvelle valeur -1
   - Ancienne valeur 1 (gris) → Nouvelle valeur 0
   - Ancienne valeur 2 (blanc) → Nouvelle valeur +1

2. Impact sur le calcul de la heatmap :
   - Ancien : heatmap = X * W
   - Nouveau : heatmap = (X - 1) * W = X_n * W

3. Conséquence cruciale :
   - Un pixel noir (ancien X=0) a maintenant X_n = -1
   - Sa contribution à la heatmap devient : -1 * W[i] = -W[i]
   - Le poids n'est plus masqué ! Il est maintenant visible avec le signe approprié

4. Avantages de cette normalisation :
   a) COMPLÉTITÉ DE L'INFORMATION :
      - Tous les poids sont maintenant visibles dans la heatmap
      - Les zones d'arrière-plan (noires) montrent leur contribution réelle
   
   b) INTERPRÉTATION AMÉLIORÉE :
      - Les pixels noirs (X_n = -1) avec poids positifs contribuent négativement
        → Logique : l'arrière-plan ne devrait pas soutenir la classe
      - Les pixels noirs avec poids négatifs contribuent positivement
        → Logique : l'absence d'encre à cet endroit soutient la classe
   
   c) SYMÉTRIE DES DONNÉES :
      - Les données sont centrées autour de 0 (moyenne ≈ 0)
      - Cela améliore la stabilité numérique des algorithmes d'optimisation
      - Les gradients sont mieux équilibrés pendant l'entraînement

   d) MEILLEURE ANALYSE DES ERREURS :
      - On peut maintenant voir si une erreur provient de poids importants
        dans les zones d'arrière-plan
      - L'analyse des heatmaps est plus fidèle au comportement réel du modèle

EFFET SUR LES PERFORMANCES :
----------------------------
Mathématiquement, pour la régression logistique :
- Ancien modèle : P(y|x) = σ(w·x + b)
- Nouveau modèle : P(y|x) = σ(w·(x-1) + b) = σ(w·x + (b - w·1))

La normalisation équivaut à un ajustement du biais : b' = b - w·1
En pratique, cela n'affecte pas significativement les performances de classification
(comme le montrent les résultats similaires avant/après normalisation),
mais améliore considérablement l'interprétabilité du modèle.
"""

# Exercice 5
def pred_lr_multi_class(X, W, B):
    """
    Prédiction pour régression logistique multi-classe avec softmax.
    
    Args:
        X: Données d'entrée (N, d)
        W: Matrice de poids (d, K)
        B: Vecteur de biais (K,)
        
    Returns:
        Matrice des probabilités (N, K)
    """
    scores = X @ W + B  # (N, K)
    # Softmax pour obtenir des probabilités
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # stabilité numérique
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
def to_categorical(Y, K):
    """
    Transforme les labels en encodage one-hot.
    
    Args:
        Y: Labels (N,)
        K: Nombre de classes
        
    Returns:
        Y_cat: Matrice one-hot (N, K)
    """
    N = len(Y)
    Y_cat = np.zeros((N, K))
    Y_cat[np.arange(N), Y.astype(int)] = 1
    return Y_cat

def rl_gradient_ascent_multi_class(X, Y, eta = 0.2, numEp = 1000, verbose = 1):
    """
    Montée de gradient pour régression logistique multi-classe (batch complet).
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels (N,)
        eta: Taux d'apprentissage
        numEp: Nombre d'époques
        verbose: Fréquence d'affichage (si verbose > 0, affiche tous les verbose epochs)
        
    Returns:
        W: Matrice de poids (d, K)
        B: Vecteur de biais (K,)
    """
    N, d = X.shape
    classes = np.unique(Y)
    K = len(classes)
    
    # Initialisation des paramètre
    W = np.zeros((d,K))
    B = np.zeros(K)
    
    # Transformation one-hot des labels
    Y_cat = to_categorical(Y, K)

    print_interval = 100 if verbose == 1 else verbose
    
    for epoch in range(numEp):
        # Prédiction
        Y_pred = pred_lr_multi_class(X, W, B)
        
        # Calcul du gradient
        error = Y_cat - Y_pred
        grad_W = X.T @ error / N
        grad_B = np.sum(error, axis=0) / N
        
        # Mise à jour des paramètres
        W += eta * grad_W
        B += eta * grad_B
        
        # Affichage périodique
        if verbose and (epoch % print_interval == 0 or epoch == numEp - 1):
            Y_pred_class = classif_multi_class(Y_pred)
            acc = accuracy(Y_pred_class, Y)
            print(f"epoch {epoch} accuracy train={acc*100:.2f} %")
    
    return W, B

def rl_gradient_ascent_multi_class_batch(X, Y, tbatch, eta = 0.2, numEp = 200, verbose = 1):
    """
    Descente de gradient stochastique pour régression logistique multi-classe.
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels (N,)
        batch_size: Taille des mini-batches
        eta: Taux d'apprentissage
        numEp: Nombre d'époques
        verbose: Fréquence d'affichage
        
    Returns:
        W: Matrice de poids (d, K)
        B: Vecteur de biais (K,)
    """
    N, d = X.shape
    classes = np.unique(Y)
    K = len(classes)
    
    # Initialisation des paramètres
    W = np.zeros((d, K))
    B = np.zeros(K)
    
    # Transformation one-hot des labels
    Y_cat = to_categorical(Y, K)
    
    print_interval = 20
    
    for epoch in range(numEp):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_cat_shuffled = Y_cat[indices]
        Y_shuffled = Y[indices]
        
        # Parcours par mini-batches
        for i in range(0, N, tbatch):
            end_idx = min(i + tbatch, N)
            X_batch = X_shuffled[i:end_idx]
            Y_batch = Y_cat_shuffled[i:end_idx]
            
            # Prédiction sur le batch
            Y_pred_batch = pred_lr_multi_class(X_batch, W, B)
            
            # Calcul du gradient
            error = Y_batch - Y_pred_batch
            grad_W = X_batch.T @ error / len(X_batch)
            grad_B = np.sum(error, axis=0) / len(X_batch)
            
            # Mise à jour des paramètres
            W += eta * grad_W
            B += eta * grad_B
        
        # Affichage périodique
        if verbose and (epoch % print_interval == 0 or epoch == numEp - 1):
            Y_pred = pred_lr_multi_class(X, W, B)
            Y_pred_class = classif_multi_class(Y_pred)
            acc = accuracy(Y_pred_class, Y)
            print(f"epoch {epoch} accuracy train={acc*100:.2f} %")
    
    return W, B

def add_random_column(X, d, sigma = 1.0):
    """
    Ajoute des colonnes de bruit gaussien aux données.
    
    Args:
        X: Données originales (N, d_orig)
        d: Nombre de colonnes de bruit à ajouter
        sigma: Écart-type du bruit
        
    Returns:
        X_new: Données avec bruit ajouté (N, d_orig + d)
    """
    noise = np.random.randn(len(X), d) * sigma
    return np.hstack((X, noise))

# Exercice 6
def dimensionality_curse(X, Y, Xt, Yt):
    """
    Démonstration de la malédiction de la dimensionnalité en ajoutant du bruit.
    
    Cette fonction montre comment l'ajout de dimensions fantômes (bruit) affecte
    les performances du modèle de régression logistique multi-classe.
    
    Args:
        X: Données d'entraînement originales (N, d)
        Y: Labels d'entraînement (N,)
        Xt: Données de test originales (Nt, d)
        Yt: Labels de test (Nt,)
        
    Returns:
        acc_train_list: Liste des taux de bonne classification en entraînement
        acc_test_list: Liste des taux de bonne classification en test
    """
    # Dimensions de bruit à tester
    noise_dims = [0, 100, 200, 400, 1000]
    
    acc_train_list = []
    acc_test_list = []
    
    print("=== DIMENSIONALITÉ FANTÔME - SANS RÉGULARISATION ===")
    
    for d_noise in noise_dims:
        # Ajout des colonnes de bruit
        X_noisy = add_random_column(X, d_noise, sigma=1.0)
        Xt_noisy = add_random_column(Xt, d_noise, sigma=1.0)
        
        # Entraînement du modèle sur les données avec bruit
        W, B = rl_gradient_ascent_multi_class_batch(
            X_noisy, Y, tbatch=500, eta=0.2, numEp=200, verbose=0
        )
        
        # Prédictions
        Y_pred = classif_multi_class(pred_lr_multi_class(X_noisy, W, B))
        Yt_pred = classif_multi_class(pred_lr_multi_class(Xt_noisy, W, B))
        
        # Calcul des performances
        acc_train = accuracy(Y_pred, Y) * 100
        acc_test = accuracy(Yt_pred, Yt) * 100
        
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        
        print(f"Noise {d_noise} ‑ Tx de bonne classification en App = {acc_train:.2f} %")
        print(f"Noise {d_noise} ‑ Tx de bonne classification en Test = {acc_test:.2f} %")
    
    # Visualisation des résultats
    plot_curse_results(noise_dims, acc_train_list, acc_test_list, "Sans régularisation")
    
    return acc_train_list, acc_test_list

def rl_gradient_ascent_multi_class_batch_regul(X, Y, tbatch, eta=0.2, numEp=200, 
                                               type='l2', llamdba=0.001, verbose=0):
    """
    Descente de gradient stochastique avec régularisation.
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels (N,)
        tbatch: Taille des mini-batches
        eta: Taux d'apprentissage
        numEp: Nombre d'époques
        reg_type: Type de régularisation ('l1' ou 'l2')
        reg_lambda: Paramètre de régularisation
        verbose: Fréquence d'affichage
        
    Returns:
        W: Matrice de poids (d, K)
        B: Vecteur de biais (K,)
    """
    N, d = X.shape
    classes = np.unique(Y)
    K = len(classes)
    
    # Initialisation des paramètres
    W = np.zeros((d, K))
    B = np.zeros(K)
    
    # Transformation one-hot des labels
    Y_cat = to_categorical(Y, K)
    
    for epoch in range(numEp):
        # Mélange des données
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_cat_shuffled = Y_cat[indices]
        
        # Parcours par mini-batches
        for i in range(0, N, tbatch):
            end_idx = min(i + tbatch, N)
            X_batch = X_shuffled[i:end_idx]
            Y_batch = Y_cat_shuffled[i:end_idx]
            
            # Prédiction sur le batch
            Y_pred_batch = pred_lr_multi_class(X_batch, W, B)
            
            # Calcul du gradient original
            error = Y_batch - Y_pred_batch
            grad_W = X_batch.T @ error / len(X_batch)
            grad_B = np.sum(error, axis=0) / len(X_batch)
            
            # Ajout du terme de régularisation
            if type.lower() == 'l2':
                # L2: gradient = 2 * lambda * W
                grad_W -= 2 * llamdba * W
            elif type.lower() == 'l1':
                # L1: gradient = lambda * sign(W)
                grad_W -= llamdba * np.sign(W)
            
            # Mise à jour des paramètres
            W += eta * grad_W
            B += eta * grad_B
        
        # Affichage périodique
        if verbose > 0 and (epoch % verbose == 0 or epoch == numEp - 1):
            Y_pred = pred_lr_multi_class(X, W, B)
            Y_pred_class = classif_multi_class(Y_pred)
            acc = accuracy(Y_pred_class, Y)
            print(f"epoch {epoch} accuracy train={acc*100:.2f} %")
    
    return W, B

def dimensionality_curse_regul(X, Y, Xt, Yt, type='l2', llamdba=0.001):
    """
    Démonstration de l'effet de la régularisation face à la malédiction de la dimensionnalité.
    
    Cette fonction montre comment la régularisation (L1 ou L2) peut atténuer
    les effets négatifs de l'ajout de dimensions fantômes.
    
    Args:
        X: Données d'entraînement originales (N, d)
        Y: Labels d'entraînement (N,)
        Xt: Données de test originales (Nt, d)
        Yt: Labels de test (Nt,)
        reg_type: Type de régularisation ('l1' ou 'l2')
        reg_lambda: Paramètre de régularisation
        
    Returns:
        acc_train_list: Liste des taux de bonne classification en entraînement
        acc_test_list: Liste des taux de bonne classification en test
    """
    # Dimensions de bruit à tester
    noise_dims = [0, 100, 200, 400, 1000]
    
    acc_train_list = []
    acc_test_list = []
    
    print(f"=== DIMENSIONALITÉ FANTÔME - AVEC RÉGULARISATION {type.upper()} (λ={llamdba}) ===")
    
    for d_noise in noise_dims:
        # Ajout des colonnes de bruit
        X_noisy = add_random_column(X, d_noise, sigma=1.0)
        Xt_noisy = add_random_column(Xt, d_noise, sigma=1.0)
        
        # Entraînement du modèle avec régularisation
        W, B = rl_gradient_ascent_multi_class_batch_regul(
            X_noisy, Y, tbatch=500, eta=0.2, numEp=200,
            type=type, llamdba=llamdba, verbose=0
        )
        
        # Prédictions
        Y_pred = classif_multi_class(pred_lr_multi_class(X_noisy, W, B))
        Yt_pred = classif_multi_class(pred_lr_multi_class(Xt_noisy, W, B))
        
        # Calcul des performances
        acc_train = accuracy(Y_pred, Y) * 100
        acc_test = accuracy(Yt_pred, Yt) * 100
        
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        
        print(f"Noise {d_noise} ‑ Tx de bonne classification en App = {acc_train:.2f} %")
        print(f"Noise {d_noise} ‑ Tx de bonne classification en Test = {acc_test:.2f} %")
    
    # Visualisation des résultats
    plot_curse_results(noise_dims, acc_train_list, acc_test_list, 
                      f"Avec régularisation {type.upper()} (λ={llamdba})")
    
    return acc_train_list, acc_test_list

def plot_curse_results(noise_dims, acc_train, acc_test, title):
    """
    Visualise les résultats de la malédiction de la dimensionnalité.
    
    Args:
        noise_dims: Liste des dimensions de bruit
        acc_train: Liste des taux de bonne classification en entraînement
        acc_test: Liste des taux de bonne classification en test
        title: Titre du graphique
    """
    plt.figure(figsize=(10, 6))
    
    # Courbe d'apprentissage (rouge)
    plt.plot(noise_dims, acc_train, 'o-', color='red', linewidth=2, markersize=8,
             label='Apprentissage (Train)')
    
    # Courbe de test (bleue)
    plt.plot(noise_dims, acc_test, 's-', color='blue', linewidth=2, markersize=8,
             label='Test')
    
    plt.xlabel('Nombre de dimensions fantômes ajoutées', fontsize=12)
    plt.ylabel('Taux de bonne classification (%)', fontsize=12)
    plt.title(f'Malédiction de la dimensionnalité: {title}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Ajouter des annotations pour les points
    for i, dim in enumerate(noise_dims):
        plt.annotate(f'{acc_train[i]:.1f}%', 
                     (dim, acc_train[i]), 
                     textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=10, color='red')
        plt.annotate(f'{acc_test[i]:.1f}%', 
                     (dim, acc_test[i]), 
                     textcoords="offset points",
                     xytext=(0,-15), ha='center', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.show()

"""
Exercice 6 : Analyse des résultats de régularisation pour la régression logistique avec dimensions fantômes

Question : Expliquer quelle régularisation est la plus adaptée ici.

Réponse :
D'après les résultats expérimentaux, la régularisation L1 (λ=0.001) est la plus adaptée 
pour ce problème de classification avec dimensions fantômes (bruit ajouté).

RATIONNEL DÉTAILLÉ :

1. PERFORMANCE DE GÉNÉRALISATION (TEST) :
   ---------------------------------------------------------------
   Sans régularisation :
   - Test chute rapidement : 94.30% → 88.63% (-5.67 points)
   - Sur-apprentissage marqué : 97.51% → 100% en Apprentissage
   
   Avec L2 (λ=0.001) :
   - Test chute également : 93.97% → 87.68% (-6.29 points)
   - Performance légèrement inférieure à L1
   
   Avec L1 (λ=0.001) :
   - Test plus stable : 93.32% → 90.94% (-2.38 points seulement)
   - Meilleure résistance au bruit, surtout avec 1000 dimensions fantômes

2. COMPORTEMENT FACE AU BRUIT CROISSANT :
   ---------------------------------------------------------------
   • L1 maintient une différence Apprentissage/Test plus raisonnable
     Ex: 1000 dimensions fantômes : 99.25% vs 90.94% (écart 8.31 points)
   • Sans régularisation : 100% vs 88.63% (écart 11.37 points)
     → Sur-apprentissage extrême
   • L2 : 100% vs 87.68% (écart 12.32 points)
     → Sur-apprentissage encore plus prononcé

3. PROPRIÉTÉS THÉORIQUES EXPLIQUANT CES RÉSULTATS :
   ---------------------------------------------------------------
   a) RÉGULARISATION L1 (LASSO) :
      - Crée de la SPARSITÉ : met exactement à 0 les poids des features inutiles
      - Sélection automatique de features : ignore complètement les dimensions fantômes
      - Particulièrement adaptée quand beaucoup de features sont du bruit
   
   b) RÉGULARISATION L2 (RIDGE) :
      - Réduit l'amplitude des poids, mais ne les annule pas
      - Tous les features (y compris bruit) contribuent un peu
      - Moins efficace quand beaucoup de dimensions sont purement du bruit
   
   c) SANS RÉGULARISATION :
      - Le modèle utilise TOUTES les features, même le bruit, pour atteindre 100% sur App
      - Apprend par cœur le bruit spécifique à l'ensemble d'apprentissage
      - Généralisation médiocre car le bruit test est différent

4. OPTIMALITÉ DE L1 DANS CE CONTEXTE :
   ---------------------------------------------------------------
   Les dimensions fantômes sont par définition :
   - Non informatives (bruit pur)
   - Non corrélées avec la cible
   - Nombreuses (jusqu'à 1000 vs 256 features originaux)

   L1 excelle dans ce scénario car :
   1. Elle identifie et élimine les features non informatifs
   2. Elle préserve les features originaux importants pour les chiffres
   3. Elle empêche le modèle de s'appuyer sur des corrélations aléatoires
   4. Elle produit un modèle plus interprétable (moins de paramètres non nuls)

5. COMPROMIS BIAS-VARIANCE :
   ---------------------------------------------------------------
   • Sans régularisation : Variance élevée, faible biais → Sur-apprentissage
   • L2 : Réduit la variance modérément
   • L1 : Réduit fortement la variance en augmentant légèrement le biais
          → Meilleur compromis pour les données bruitées

RECOMMANDATIONS PRATIQUES :
---------------------------
1. Pour des problèmes avec potentiellement beaucoup de features non informatifs, 
   préférer L1.
2. Ajuster λ par validation croisée (ici λ=0.001 donne de bons résultats).
3. Combiner L1 et L2 (Elastic Net) si certaines corrélations entre features 
   sont importantes.
4. Toujours comparer avec et sans régularisation pour quantifier le bénéfice.

REMARQUE FINALE :
-----------------
Les résultats montrent clairement que sans régularisation, le modèle devient 
complètement non généralisable avec beaucoup de bruit (100% App vs 88.63% Test).
La régularisation L1 atténue fortement ce problème, prouvant son utilité cruciale 
dans les scénarios réalistes où les données contiennent toujours une certaine 
proportion de features non informatifs.
"""

# Exercice 7
def perceptron_binaire(X, Y, eta=0.1, epochs=1000, verbose=0):
    """
    Perceptron pour classification binaire avec labels {-1, 1}.
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels binaires (N,) ∈ {-1, 1}
        eta: Taux d'apprentissage
        epochs: Nombre maximum d'époques
        verbose: Niveau d'affichage (0: silencieux, 1: affiche la progression)
        
    Returns:
        w: Vecteur de poids (d,)
        b: Biais (scalaire)
        history: Historique des erreurs par époque (optionnel)
    """
    N, d = X.shape
    
    # Initialisation
    w = np.zeros(d)
    b = 0
    history = []
    
    for epoch in range(epochs):
        errors = 0
        # Parcours des données dans l'ordre
        for i in range(N):
            xi = X[i]
            yi = Y[i]
            
            # Condition de mise à jour du perceptron
            if yi * (np.dot(w, xi) + b) <= 0:
                w += eta * yi * xi
                b += eta * yi
                errors += 1
        
        history.append(errors)
        
        # Affichage de la progression
        if verbose > 0 and epoch % 100 == 0:
            print(f"Époque {epoch}: erreurs = {errors}")
        
        # Arrêt prématuré si aucune erreur
        if errors == 0:
            if verbose > 0:
                print(f"Convergence atteinte à l'époque {epoch}")
            break
    
    return w, b, np.array(history)
    
def to_binary_labels(Y, target_class):
    """
    Transforme les labels multi-classes en labels binaires {-1, 1} pour la classe cible.
    
    Args:
        Y: Labels multi-classes (N,) ∈ {0, 1, ..., K-1}
        target_class: Classe à considérer comme positive (1)
        
    Returns:
        Y_bin: Labels binaires (N,) ∈ {-1, 1}
    """
    return np.where(Y == target_class, 1, -1)
    
def perceptron_one_against_all(X, Y, eta=0.1, epochs=1000, verbose=0):
    """
    Perceptron multi-classes utilisant la stratégie un-contre-tous.
    
    Args:
        X: Données d'entraînement (N, d)
        Y: Labels multi-classes (N,) ∈ {0, 1, ..., K-1}
        eta: Taux d'apprentissage
        epochs: Nombre maximum d'époques
        verbose: Niveau d'affichage
        
    Returns:
        W: Matrice de poids (d, K), chaque colonne correspond à un classifieur binaire
        B: Vecteur de biais (K,)
        histories: Liste des historiques d'erreurs pour chaque classifieur
    """
    N, d = X.shape
    classes = np.unique(Y)
    K = len(classes)
    
    # Initialisation
    W = np.zeros((d, K))
    B = np.zeros(K)
    histories = []
    
    print(f"Entraînement du Perceptron multi-classes (stratégie un-contre-tous)")
    print(f"Nombre de classes: {K}")
    print(f"Nombre de données: {N}")
    
    # Entraînement d'un classifieur binaire pour chaque classe
    for idx, cl in enumerate(classes):
        if verbose > 0:
            print(f"\nEntraînement du classifieur pour la classe {cl}...")
        
        # Transformation des labels en binaire {-1, 1}
        Y_bin = to_binary_labels(Y, cl)
        
        # Entraînement du perceptron binaire
        w_cl, b_cl, history_cl = perceptron_binaire(
            X, Y_bin, eta=eta, epochs=epochs, verbose=verbose-1 if verbose > 1 else 0
        )
        
        # Stockage des paramètres
        W[:, idx] = w_cl
        B[idx] = b_cl
        histories.append(history_cl)
        
        # Calcul de l'accuracy sur les données d'entraînement
        predictions = np.sign(X.dot(w_cl) + b_cl)
        accuracy = np.mean(predictions == Y_bin)
        
        if verbose > 0:
            print(f"  Classe {cl}: accuracy = {accuracy*100:.2f}%, "
                    f"poids non-nuls = {np.sum(np.abs(w_cl) > 1e-6)}/{d}")
    
    return W, B, histories

def predict_perceptron_one_against_all(X, W, B):
    """
    Prédiction multi-classes avec les classifieurs Perceptron un-contre-tous.
    
    Args:
        X: Données à prédire (M, d)
        W: Matrice de poids (d, K)
        B: Vecteur de biais (K,)
        
    Returns:
        predictions: Labels prédits (M,)
        scores: Scores bruts (M, K)
    """
    # Calcul des scores pour chaque classifieur
    scores = X.dot(W) + B  # (M, K)
    
    # Pour chaque échantillon, choisir la classe avec le score le plus élevé
    predictions = np.argmax(scores, axis=1)
    
    return predictions, scores

"""
Exercice 7: Test de l'algorithme Perceptron et comparaison avec la régression logistique

Analyse des résultats obtenus:

1. PERFORMANCE DU PERCEPTRON BINAIRE (chiffre 0 vs tous):
   ------------------------------------------------------
   - Précision en apprentissage: 99.44%
   - Précision en test: 98.27%
   - Le Perceptron converge rapidement (231 erreurs à la première époque)
   - Excellente séparation linéaire pour le chiffre 0
   - Très peu de sur-apprentissage (écart de seulement 1.17%)

2. PERFORMANCE DU PERCEPTRON MULTICLASSE:
   ---------------------------------------
   - Précision en apprentissage: 96.77%
   - Précision en test: 91.59%
   - Écart apprentissage/test: 5.18% → signe de sur-apprentissage modéré
   - Toutes les classes obtiennent plus de 97.5% en apprentissage
   - Les poids sont presque tous non-nuls (256/256 pour la plupart)

3. ANALYSE DES ERREURS:
   --------------------
   - 258 erreurs sur 3069 échantillons de test (8.41%)
   - Confusions fréquentes: 3→5 (17), 4→2 (17), 8→2 (14)
   - Ces confusions sont attendues (chiffres visuellement similaires)
   - Le Perceptron a du mal avec les formes ambigües

4. COMPARAISON AVEC LA RÉGRESSION LOGISTIQUE:
   -------------------------------------------
   - Perceptron: 96.77% apprentissage, 91.59% test
   - Régression logistique: 96.13% apprentissage, 93.74% test
   - La régression logistique généralise mieux (+2.15% en test)
   - Le Perceptron sur-apprend davantage (écart plus grand)

Interprétation:
---------------
1. Le Perceptron est efficace pour des problèmes linéairement séparables
   comme la reconnaissance du chiffre 0.
   
2. Pour le multi-classe, la performance baisse car certaines paires de chiffres
   ne sont pas linéairement séparables dans l'espace des pixels bruts.
   
3. La régression logistique, en optimisant une fonction de coût probabiliste,
   obtient de meilleures performances de généralisation.

4. Les confusions fréquentes (3/5, 4/2, 8/2) sont cohérentes avec la difficulté
   visuelle à distinguer ces chiffres.
"""

"""
Spécificités du Perceptron:
---------------------------
- Pour le Perceptron, le codage des classes est en {-1, 1}
- C'est un algorithme d'apprentissage en ligne (online)
- Fonctionne bien pour les problèmes linéairement séparables
- Pas de fonction de coût explicite, mais plutôt une condition de marge
- Convergence rapide pour les données séparables linéairement
- Peut ne pas converger pour les données non séparables

Comparaison avec la régression logistique:
------------------------------------------
Perceptron:
- Avantages: Simple, rapide, peu de calculs par itération
- Mise à jour uniquement lorsque l'exemple est mal classé
- Pas de notion de probabilité, décision binaire
- Pas de garantie de convergence pour les données non séparables
- Sensible à l'ordre de présentation des données

Régression logistique:
- Avantages: Optimise une fonction de coût (entropie croisée)
- Donne des probabilités d'appartenance aux classes
- Fonctionne bien même pour les données non linéairement séparables
- Moins sensible aux outliers
- Convergence garantie vers un optimum local
- Plus lourd en calculs (nécessite le gradient sur tout le dataset)

Ce script teste les deux approches sur le dataset USPS et compare leurs performances.

Résultats observés:
- Le Perceptron obtient 91.59% de précision en test
- La régression logistique obtient 93.74% de précision en test
- La régression logistique généralise mieux (+2.15%)
- Le Perceptron montre plus de sur-apprentissage (écart de 5.18% vs 2.39%)
"""

"""
# ============================
# Exercice 7: Test de l'algorithme Perceptron
# ============================

print("=" * 60)
print("Exercice 7: Test de l'algorithme Perceptron")
print("=" * 60)

# -----------------------------------------------------------------
# 1. Test du Perceptron binaire (un chiffre contre tous)
# -----------------------------------------------------------------
print("\n1. Test du Perceptron binaire (classe 0 contre toutes)")

# Conversion des labels en binaire {0, 1} → {-1, 1}
classe_cible = 0
Y_bin_apprentissage = np.where(Y == classe_cible, 1, -1)  # Classe 0: 1, autres: -1
Y_bin_test = np.where(Yt == classe_cible, 1, -1)

# Entraînement du Perceptron binaire
print(f"Entraînement du Perceptron binaire pour reconnaître le chiffre {classe_cible}...")
w, b, historique = tme9.perceptron_binaire(
    X, Y_bin_apprentissage, 
    eta=0.1,          # Taux d'apprentissage
    epochs=100,       # Nombre d'époques
    verbose=1         # Affichage de la progression
)

# Visualisation du processus d'entraînement
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(historique, 'b-', linewidth=2)
plt.xlabel('Époques (epoch)')
plt.ylabel('Nombre d\'échantillons erronés')
plt.title('Processus d\'entraînement du Perceptron (Erreurs vs Époques)')
plt.grid(True, alpha=0.3)

# Prédictions sur l'ensemble d'apprentissage
predictions_apprentissage = np.sign(X @ w + b)
precision_apprentissage = np.mean(predictions_apprentissage == Y_bin_apprentissage) * 100

# Prédictions sur l'ensemble de test
predictions_test = np.sign(Xt @ w + b)
precision_test = np.mean(predictions_test == Y_bin_test) * 100

print(f"Précision en apprentissage: {precision_apprentissage:.2f}%")
print(f"Précision en test: {precision_test:.2f}%")

# Visualisation du vecteur de poids
plt.subplot(1, 2, 2)
if len(w) == 256:
    plt.imshow(w.reshape(16, 16), cmap='gray')
    plt.colorbar()
    plt.title('Visualisation des poids du Perceptron')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# 2. Test du Perceptron multiclasse (stratégie un contre tous)
# -----------------------------------------------------------------
print("\n2. Test du Perceptron multiclasse (stratégie un contre tous)")

print("Début de l'entraînement du Perceptron multiclasse...")
W_perceptron, B_perceptron, historiques = tme9.perceptron_one_against_all(
    X, Y,
    eta=0.1,      # Taux d'apprentissage
    epochs=50,    # Nombre d'époques par classe
    verbose=1     # Affichage des informations détaillées
)

# Prédictions sur l'ensemble d'apprentissage
predictions_apprentissage_multi, scores_apprentissage = tme9.predict_perceptron_one_against_all(X, W_perceptron, B_perceptron)
precision_apprentissage_multi = tme9.accuracy(predictions_apprentissage_multi, Y) * 100

# Prédictions sur l'ensemble de test
predictions_test_multi, scores_test = tme9.predict_perceptron_one_against_all(Xt, W_perceptron, B_perceptron)
precision_test_multi = tme9.accuracy(predictions_test_multi, Yt) * 100

print(f"\nRésultats du Perceptron multiclasse:")
print(f"Précision en apprentissage: {precision_apprentissage_multi:.2f}%")
print(f"Précision en test: {precision_test_multi:.2f}%")

# -----------------------------------------------------------------
# 3. Visualisation des poids pour chaque classe
# -----------------------------------------------------------------
print("\n3. Visualisation des poids du Perceptron pour chaque classe")

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    if W_perceptron.shape[0] == 256:
        plt.imshow(W_perceptron[:, i].reshape(16, 16), cmap='gray')
    plt.title(f"Chiffre {i}")
    plt.axis('off')
plt.suptitle('Perceptron: Visualisation des poids pour chaque chiffre', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# 4. Analyse des échantillons mal classés
# -----------------------------------------------------------------
print("\n4. Analyse des échantillons mal classés")

# Identification des échantillons mal classés dans le test
indices_erreurs = np.where(predictions_test_multi != Yt)[0]
nombre_erreurs = len(indices_erreurs)

print(f"Nombre d'erreurs de classification en test: {nombre_erreurs}/{len(Yt)} ({nombre_erreurs/len(Yt)*100:.2f}%)")

if nombre_erreurs > 0:
    # Affichage des premiers échantillons mal classés
    nombre_a_afficher = min(5, nombre_erreurs)
    print(f"\nAffichage des {nombre_a_afficher} premières erreurs de classification:")
    
    plt.figure(figsize=(15, 3 * nombre_a_afficher))
    for i, idx in enumerate(indices_erreurs[:nombre_a_afficher]):
        vrai_label = Yt[idx]
        label_prediction = predictions_test_multi[idx]
        
        plt.subplot(nombre_a_afficher, 1, i + 1)
        plt.imshow(Xt[idx].reshape(16, 16), cmap='gray')
        plt.title(f"Échantillon {idx}: Vrai label={vrai_label}, Prédiction={label_prediction}", 
                 color='red' if vrai_label != label_prediction else 'green')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analyse des types d'erreurs les plus fréquents
    print("\nTypes d'erreurs les plus fréquents:")
    compteurs_erreurs = {}
    for idx in indices_erreurs:
        vrai_label = Yt[idx]
        label_prediction = predictions_test_multi[idx]
        type_erreur = f"{vrai_label}→{label_prediction}"
        compteurs_erreurs[type_erreur] = compteurs_erreurs.get(type_erreur, 0) + 1
    
    # Tri par fréquence décroissante
    erreurs_triees = sorted(compteurs_erreurs.items(), key=lambda x: x[1], reverse=True)
    for type_erreur, compte in erreurs_triees[:10]:
        print(f"  {type_erreur}: {compte} fois")

# -----------------------------------------------------------------
# 5. Comparaison des performances: Perceptron vs Régression logistique
# -----------------------------------------------------------------
print("\n5. Comparaison des performances: Perceptron vs Régression logistique")

# Ré-entraînement de la régression logistique comme référence (avec les fonctions précédentes)
print("Entraînement de la régression logistique comme référence...")
W_rl, B_rl = tme9.rl_gradient_ascent_multi_class_batch(
    X, Y, tbatch=500, eta=0.2, numEp=50, verbose=0
)

# Prédictions de la régression logistique
predictions_apprentissage_rl = tme9.classif_multi_class(tme9.pred_lr_multi_class(X, W_rl, B_rl))
predictions_test_rl = tme9.classif_multi_class(tme9.pred_lr_multi_class(Xt, W_rl, B_rl))

precision_apprentissage_rl = tme9.accuracy(predictions_apprentissage_rl, Y) * 100
precision_test_rl = tme9.accuracy(predictions_test_rl, Yt) * 100

print("\nTableau de comparaison des performances:")
print("-" * 50)
print(f"{'Méthode':<20} {'Précision Apprentissage':<15} {'Précision Test':<15}")
print("-" * 50)
print(f"{'Perceptron':<20} {precision_apprentissage_multi:<15.2f} {precision_test_multi:<15.2f}")
print(f"{'Régression logistique':<20} {precision_apprentissage_rl:<15.2f} {precision_test_rl:<15.2f}")
print("-" * 50)

# Visualisation comparative
plt.figure(figsize=(10, 5))
methodes = ['Perceptron', 'Régression logistique']
precisions_apprentissage = [precision_apprentissage_multi, precision_apprentissage_rl]
precisions_test = [precision_test_multi, precision_test_rl]

x = np.arange(len(methodes))
largeur = 0.35

plt.bar(x - largeur/2, precisions_apprentissage, largeur, label='Précision Apprentissage', color='skyblue')
plt.bar(x + largeur/2, precisions_test, largeur, label='Précision Test', color='lightcoral')

plt.xlabel('Méthode')
plt.ylabel('Précision (%)')
plt.title('Comparaison des performances: Perceptron vs Régression logistique')
plt.xticks(x, methodes)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Ajout des valeurs sur les barres
for i, (apprentissage, test) in enumerate(zip(precisions_apprentissage, precisions_test)):
    plt.text(i - largeur/2, apprentissage + 0.5, f'{apprentissage:.1f}%', ha='center', va='bottom')
    plt.text(i + largeur/2, test + 0.5, f'{test:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
"""

"""
Test complet de l'algorithme Perceptron

Ce script évalue le Perceptron selon plusieurs aspects:
1. Performance en classification binaire (un chiffre contre tous)
2. Performance en classification multi-classe (stratégie un contre tous)
3. Visualisation des poids appris
4. Analyse des erreurs de classification
5. Comparaison avec la régression logistique

Méthodologie:
-------------
1. Pour chaque classe (0-9), on entraîne un classifieur Perceptron binaire
2. Chaque classifieur apprend à séparer sa classe cible des autres
3. Pour la prédiction multi-classe, on choisit la classe avec le score le plus élevé
4. On évalue les performances sur les ensembles d'apprentissage et de test
"""