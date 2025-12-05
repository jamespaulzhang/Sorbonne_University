# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
def crossval(X, Y, n_iterations, iteration):
    N = len(X)
    fold_size = N // n_iterations

    i_start = iteration * fold_size
    i_end = (iteration + 1) * fold_size

    Xtest = X[i_start:i_end]
    Ytest = Y[i_start:i_end]

    Xapp = np.concatenate((X[:i_start], X[i_end:]), axis=0)
    Yapp = np.concatenate((Y[:i_start], Y[i_end:]), axis=0)

    return Xapp, Yapp, Xtest, Ytest

# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    unique_classes, class_counts = np.unique(Y, return_counts=True)
    indices_by_class = {cls: np.where(Y == cls)[0] for cls in unique_classes}
    
    test_indices = []
    train_indices_set = set(range(X.shape[0])) # On part de tous les indices
    
    for cls in unique_classes:
        indices = indices_by_class[cls]
        fold_size = len(indices) // n_iterations
        i_start = iteration * fold_size
        i_end = (iteration + 1) * fold_size if iteration < n_iterations else len(indices)
        
        test_indices.extend(indices[i_start:i_end])
    
    # Suppression des indices test pour obtenir les indices train
    train_indices = sorted(train_indices_set - set(test_indices))

    Xtest, Ytest = X[test_indices], Y[test_indices]
    Xapp, Yapp = X[train_indices], Y[train_indices]
    
    return Xapp, Yapp, Xtest, Ytest

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = np.mean(L)
    ecart_type = np.std(L)
    return (moyenne, ecart_type)

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS
    perf = []
    
    print("------ affichage validation croisée (optionnel)")
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        classifieur = copy.deepcopy(C)
        classifieur.train(Xapp, Yapp)
        Y_pred = np.array([classifieur.predict(x) for x in Xtest])
        taux_bonne_classification = np.mean(Y_pred == Ytest)
        perf.append(taux_bonne_classification)
        print(f'Itération {i}: taille base app.= {Xapp.shape[0]}\ttaille base test={Xtest.shape[0]}\tTaux de bonne classif: {taux_bonne_classification:.4f}')
        
    taux_moyen = np.mean(perf)
    taux_ecart = np.std(perf)
    
    print("------ fin affichage validation croisée")
    return perf, taux_moyen, taux_ecart

def xie_beni_index_manual(X, labels, centroids):
    n_samples = X.shape[0]
    # Somme des distances intra-cluster au carré
    intra_distances = 0.0
    for i in range(n_samples):
        xi = X[i]
        ci = centroids[labels[i]]
        intra_distances += np.sum((xi - ci) ** 2)

    # Distance minimale inter-cluster (entre centroïdes)
    min_intercluster_dist_sq = float('inf')
    k = len(centroids)
    for i in range(k):
        for j in range(i + 1, k):
            dist_sq = np.sum((centroids[i] - centroids[j]) ** 2)
            if dist_sq < min_intercluster_dist_sq:
                min_intercluster_dist_sq = dist_sq

    # Calcul de l'indice Xie-Beni
    xb = intra_distances / (n_samples * min_intercluster_dist_sq)
    return xb

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def dunn_index_manual(X, labels):
    unique_labels = np.unique(labels)
    clusters = [X[labels == l] for l in unique_labels]
    
    # 1. Distance minimale entre deux clusters
    min_intercluster = float('inf')
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for a in clusters[i]:
                for b in clusters[j]:
                    d = euclidean_dist(a, b)
                    if d < min_intercluster:
                        min_intercluster = d
    
    # 2. Distance maximale à l'intérieur d’un cluster
    max_intracluster = 0.0
    for cluster in clusters:
        n = len(cluster)
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean_dist(cluster[i], cluster[j])
                if d > max_intracluster:
                    max_intracluster = d
    
    # Calcul de l’indice de Dunn
    if max_intracluster == 0:
        return 0
    return min_intercluster / max_intracluster
# ------------------------ 