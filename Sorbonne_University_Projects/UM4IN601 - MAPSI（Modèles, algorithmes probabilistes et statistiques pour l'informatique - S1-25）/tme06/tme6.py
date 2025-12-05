# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import matplotlib.pyplot as plt
import numpy as np

def discretise(X, n_etats):
    """
    Discrétise une liste de signaux d'angles en un nombre fini d'états.
    
    Args:
        X : liste de signaux (chaque signal est un np.array d'angles en degrés)
        n_etats : nombre d'états souhaité (ex: 20)
    
    Retourne:
        liste de signaux discrétisés (même structure que X)
    """
    intervalle = 360 / n_etats
    Xd = [np.floor(x / intervalle) for x in X]
    return Xd

def groupByLabel(Y):
    """
    Regroupe les indices des signaux par classe (lettre).

    Args:
        Y : liste ou array de labels (chaque élément est une lettre)
    
    Returns:
        dict : {lettre: np.array(indices correspondants)}
    """
    groups = {}
    for idx, label in enumerate(Y):
        if label not in groups:
            groups[label] = []
        groups[label].append(idx)
    
    # Convertir les listes en np.array
    for label in groups:
        groups[label] = np.array(groups[label])
    
    return groups

def learnMarkovModel(X, d):
    """
    Apprend un modèle de Markov pour une classe donnée.
    
    Args:
        X : liste de signaux
        d : nombre d'états
    
    Returns:
        (Pi, A) : tuple des distributions initiale et de transition
    """

    # Discrétisation de tous les signaux
    Xd = discretise(X, d)
    Xd = [x.astype(int) for x in Xd]

    A = np.zeros((d, d))
    Pi = np.zeros(d)
    
    for x in Xd:
        Pi[x[0]] += 1
        for t in range(len(x)-1):
            A[x[t], x[t+1]] += 1
            
    # Normalisation
    A = A / np.maximum(A.sum(1).reshape(d,1), 1)
    Pi = Pi / Pi.sum()
    
    return Pi, A

'''
La solution actuelle pour gérer les lignes entièrement à 0 dans la matrice de transition A consiste à diviser par 1 pour éviter la division par zéro :

A = A / np.maximum(A.sum(1).reshape(d,1), 1)


Problème :

Si une ligne de A est entièrement à 0, cela signifie que ce état n’a jamais été observé dans les données d’apprentissage comme état de départ pour une transition.

En divisant par 1, on obtient une ligne de 0 dans la matrice normalisée, ce qui n’est pas une distribution de probabilités valide (la somme doit être 1).

Cette approche est donc naïve et peut poser des problèmes lors de la génération ou de l’évaluation de séquences, car un état ne peut jamais passer à un autre état si sa ligne est nulle.
'''

def learn_all_MarkovModels(X, Y, d):
    """
    Apprend un modèle de Markov pour chaque classe et les stocke dans un dictionnaire.
    
    Args:
        X : liste de signaux (chaque signal est un np.array d'angles)
        Y : liste de labels correspondants à X
        d : nombre d'états pour la discrétisation
    
    Returns:
        dict : {lettre: (Pi, A)}
    """
    models = {}
    
    groups = groupByLabel(Y)
    
    for lettre, indices in groups.items():
        X_class = [X[i] for i in indices]
        Pi, A = learnMarkovModel(X_class, d)
        models[lettre] = (Pi, A)
    
    return models

def stationary_distribution_freq(Xd, d):
    # Compter toutes les occurrences des états
    counts = np.zeros(d)
    for x in Xd:
        x_int = x.astype(int)
        for s in x_int:
            counts[s] += 1
    return counts / counts.sum()

def stationary_distribution_sampling(Pi, A, N):
    d = len(Pi)
    # Choisir un état initial selon Pi
    state = np.random.choice(d, p=Pi)
    counts = np.zeros(d)
    
    for _ in range(N):
        counts[state] += 1
        state = np.random.choice(d, p=A[state])
    
    return counts / counts.sum()

def stationary_distribution_fixed_point(A, epsilon):
    d = A.shape[0]
    pi = np.ones(d) / d  # initialisation uniforme
    diff = 1.0
    while diff > epsilon:
        pi_new = pi @ A
        diff = np.square(pi_new - pi).mean()
        pi = pi_new
    return pi

def stationary_distribution_fixed_point_VP(A, epsilon=1e-10):
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigvals-1))  # valeur propre la plus proche de 1
    pi = np.real(eigvecs[:, idx])
    pi /= pi.sum()  # normalisation
    return pi

def logL_Sequence(s, Pi, A):
    """
    Calcule la log-vraisemblance d'une séquence s pour un modèle de Markov (Pi, A).
    
    Args:
        s : liste ou np.array d'états
        Pi : np.array des probabilités initiales (taille d)
        A  : np.array de transition (d x d)
    
    Returns:
        logL : float, log-vraisemblance de la séquence
    """
    s = np.array(s, dtype=int)
    logL = 0.0
    
    # premier état
    logL += np.log(Pi[s[0]])
    
    # parcours de la séquence
    for t in range(1, len(s)):
        logL += np.log(A[s[t-1], s[t]])
    
    return logL

"""
Question : Le signal Xd[0] est-il bien classé ?
Réponse : 
La classe choisie est celle qui maximise la log-vraisemblance.
Ici, la log-vraisemblance maximale est pour la classe 'z' (-12.48),
donc le signal Xd[0] sera classé dans la classe 'z'.

Question : D'où viennent tous les `-inf` ? 
Réponse : 
Les valeurs -inf apparaissent car la log-vraisemblance est calculée ainsi :
logL = log(Pi[s[0]]) + sum_{t=1}^{T-1} log(A[s[t-1], s[t]])
Si Pi[i] = 0 ou A[i,j] = 0 pour un état ou une transition dans la séquence,
alors np.log(0) = -inf.
Concrètement, cela signifie que certaines transitions ou états
n’ont jamais été observés dans les données d’apprentissage pour ces modèles.

Remarque : pour éviter les -inf, on pourrait appliquer un lissage
de type Laplace sur Pi et A, mais ce n’est pas demandé ici.
"""

def compute_all_ll(Xd, models):
    """
    Calcule la log-vraisemblance de tous les signaux pour tous les modèles.
    
    Args:
        Xd : liste de signaux discrétisés
        models : dictionnaire {lettre: (Pi, A)}
    
    Returns:
        np.array : matrice de taille (n_signaux, n_lettres) avec les log-vraisemblances
    """
    n_signaux = len(Xd)
    lettres = sorted(models.keys())  # pour avoir un ordre fixe
    n_lettres = len(lettres)
    
    ll_matrix = np.zeros((n_signaux, n_lettres))
    
    for i, signal in enumerate(Xd):
        for j, lettre in enumerate(lettres):
            Pi, A = models[lettre]
            ll_matrix[i, j] = logL_Sequence(signal, Pi, A)
    
    return ll_matrix

def accuracy(ll, Y):
    """
    Calcule le pourcentage de bonne classification.
    
    Args:
        ll : matrice des log-vraisemblances (n_signaux, n_lettres)
        Y : labels réels
    
    Returns:
        float : pourcentage de bonne classification
    """
    n_signaux = len(Y)
    lettres = sorted(set(Y))  # toutes les lettres possibles
    
    # Pour chaque signal, trouver la lettre avec la plus grande log-vraisemblance
    predictions = []
    for i in range(n_signaux):
        best_idx = np.argmax(ll[i, :])
        predictions.append(lettres[best_idx])
    
    # Compter les bonnes prédictions
    correct = sum(1 for pred, true in zip(predictions, Y) if pred == true)
    
    return correct / n_signaux

def learnMarkovModel_Laplace(X, d):
    """
    Version avec lissage de Laplace pour éviter les probabilités nulles.
    
    Args:
        X : liste de signaux
        d : nombre d'états
    
    Returns:
        (Pi, A) : tuple des distributions initiale et de transition
    """
    Xd = discretise(X, d)
    Xd = [x.astype(int) for x in Xd]

    # Initialisation avec des 1 au lieu de 0 (lissage de Laplace)
    A = np.ones((d, d))
    Pi = np.ones(d)
    
    for x in Xd:
        Pi[x[0]] += 1
        for t in range(len(x)-1):
            A[x[t], x[t+1]] += 1
            
    # Normalisation
    A = A / np.maximum(A.sum(1).reshape(d, 1), 1)
    Pi = Pi / Pi.sum()
    
    return Pi, A

def learn_all_MarkovModels_Laplace(X, Y, d):
    """
    Apprend tous les modèles avec lissage de Laplace.
    
    Args:
        X : liste de signaux
        Y : liste de labels
        d : nombre d'états
    
    Returns:
        dict : {lettre: (Pi, A)}
    """
    models = {}
    groups = groupByLabel(Y)
    
    for lettre, indices in groups.items():
        X_class = [X[i] for i in indices]
        Pi, A = learnMarkovModel_Laplace(X_class, d)
        models[lettre] = (Pi, A)
    
    return models

def confusion_matrix(ll, Y, title):
    """
    Calcule et affiche la matrice de confusion sous forme graphique.
    
    Args:
        ll : matrice des log-vraisemblances
        Y : labels réels
        title : titre pour l'affichage
    """
    lettres = sorted(set(Y))
    n_lettres = len(lettres)
    
    # Créer un mapping lettre -> index
    lettre_to_idx = {lettre: idx for idx, lettre in enumerate(lettres)}
    
    # Initialiser la matrice de confusion
    conf_matrix = np.zeros((n_lettres, n_lettres), dtype=int)
    
    # Pour chaque signal, trouver la prédiction
    for i, true_lettre in enumerate(Y):
        true_idx = lettre_to_idx[true_lettre]
        pred_idx = np.argmax(ll[i, :])
        conf_matrix[true_idx, pred_idx] += 1
    
    # Calculer l'accuracy
    accuracy_val = np.trace(conf_matrix) / np.sum(conf_matrix) * 100
    print(f"accuracy {title} = {accuracy_val}")
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer la heatmap
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
    
    # Ajouter les valeurs dans les cellules
    for i in range(n_lettres):
        for j in range(n_lettres):
            color = 'white' if conf_matrix[i, j] > np.max(conf_matrix) / 2 else 'black'
            ax.text(j, i, str(conf_matrix[i, j]), 
                    ha='center', va='center', color=color, fontsize=10)
    
    # Configurer les axes
    ax.set_xticks(np.arange(n_lettres))
    ax.set_yticks(np.arange(n_lettres))
    ax.set_xticklabels(lettres)
    ax.set_yticklabels(lettres)
    
    # Étiquettes
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')
    ax.set_title(f'Confusion {title}\nAccuracy: {accuracy_val:.2f}%')
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Nombre d\'occurrences')
    
    # Ajuster la disposition
    plt.tight_layout()
    
    # Afficher le graphique
    plt.show()
    
    return conf_matrix