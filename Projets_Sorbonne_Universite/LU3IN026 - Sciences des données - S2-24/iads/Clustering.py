# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import itertools
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------ 
def normalisation(df):
    """ Normalise un DataFrame pour que chaque colonne soit entre 0 et 1 """
    return (df - df.min()) / (df.max() - df.min())

def dist_euclidienne(v1, v2):
    """ Calcule la distance euclidienne entre deux vecteurs v1 et v2 """
    return np.sqrt(np.sum((v1 - v2) ** 2))

def centroide(data):
    """ 
    Calcule le centroïde d'un ensemble d'exemples.
    
    Paramètres:
        - data : DataFrame ou np.array contenant plusieurs exemples
    
    Retourne:
        - Un vecteur représentant le centroïde
    """
    return data.mean(axis=0)

def dist_centroides(group1, group2):
    """ 
    Calcule la distance entre deux groupes d'exemples en utilisant leurs centroïdes.
    
    Paramètres:
        - group1, group2 : DataFrame ou np.array contenant les exemples
    
    Retourne:
        - La distance euclidienne entre les centroïdes des deux groupes (float)
    """
    centroid1 = centroide(group1)
    centroid2 = centroide(group2)
    return dist_euclidienne(centroid1, centroid2)

def initialise_CHA(DF):
    """
    Initialise la partition pour le clustering hiérarchique ascendant (CHA).
    
    Paramètres:
        - DF : DataFrame représentant la base d'apprentissage.
    
    Retourne:
        - Un dictionnaire où chaque clé est un numéro d'exemple, et la valeur est une liste contenant uniquement ce numéro.
    """
    return {i: [i] for i in range(len(DF))}

def fusionne(df, P0, verbose=False):
    """
    Fusionne les 2 clusters les plus proches dans une partition selon la distance des centroïdes.

    Arguments :
    - df : DataFrame contenant les données normalisées.
    - P0 : Dictionnaire représentant la partition actuelle {clé: liste d'indices}.
    - verbose : Booléen, affiche des messages si True.

    Retourne :
    - P1 : Nouvelle partition après fusion.
    - c1, c2 : Les clés des clusters fusionnés.
    - d_min : La distance entre les deux clusters fusionnés.
    """
    # Liste des clés des clusters
    clusters = list(P0.keys())
    
    # Variables pour suivre les clusters à fusionner
    c1, c2 = None, None
    d_min = float("inf")
    
    # Recherche des 2 clusters les plus proches
    for i, j in itertools.combinations(clusters, 2):  # Toutes les paires possibles
        # Récupération des centroïdes
        centroide_i = centroide(df.iloc[P0[i]])
        centroide_j = centroide(df.iloc[P0[j]])
        
        # Calcul de la distance entre les centroïdes
        d = dist_euclidienne(centroide_i, centroide_j)
        
        # Mise à jour si une distance plus petite est trouvée
        if d < d_min:
            d_min = d
            c1, c2 = i, j

    # Affichage du premier message verbose
    if verbose:
        print(f"fusionne: distance minimale trouvée entre  [{c1}, {c2}]  =  {d_min:.16f}")

    # Fusion des deux clusters trouvés
    new_cluster = P0[c1] + P0[c2]  # Union des indices

    # Création de la nouvelle partition
    P1 = {k: v for k, v in P0.items() if k not in (c1, c2)}  # Suppression de c1 et c2
    new_key = max(P0.keys()) + 1  # Nouvelle clé pour le cluster fusionné
    P1[new_key] = new_cluster

    # Affichage des autres messages verbose
    if verbose:
        print(f"fusionne: les 2 clusters dont les clés sont  [{c1}, {c2}]  sont fusionnés")
        print(f"fusionne: on crée la nouvelle clé {new_key} dans le dictionnaire.")
        print(f"fusionne: les clés de  [{c1}, {c2}]  sont supprimées car leurs clusters ont été fusionnés.")

    return P1, c1, c2, d_min

def CHA_centroid(df, verbose=False, dendrogramme=False):
    """
    Effectue le clustering hiérarchique ascendant (CHA) en utilisant la distance des centroïdes.

    Paramètres:
        - df : DataFrame contenant les données normalisées.
        - verbose : Booléen, affiche des messages si True.
        - dendrogramme : Booléen, affiche le dendrogramme si True.

    Retourne:
        - Une liste composée de listes contenant chacune :
            - Les 2 indices d'éléments fusionnés
            - La distance les séparant
            - La somme du nombre d'éléments des 2 éléments fusionnés
    """
    # Initialisation de la partition
    P = initialise_CHA(df)

    # Liste pour stocker les résultats
    fusion_results = []

    # Matrice de liaison pour le dendrogramme
    Z = []

    # Tant qu'il y a plus d'un cluster
    while len(P) > 1:
        # Fusion des deux clusters les plus proches
        P, c1, c2, d_min = fusionne(df, P, verbose=verbose)

        # Ajout des résultats de la fusion à la liste
        fusion_results.append([c1, c2, d_min, len(P[max(P.keys())])])

        # Ajout de l'information de fusion pour le dendrogramme
        Z.append([c1, c2, d_min, len(P[max(P.keys())])])

        # Affichage des messages verbose
        if verbose:
            print(f"CHA_centroid: Fusion des clusters {c1} et {c2} à une distance de {d_min:.16f}")
            print(f"CHA_centroid: Nouveau cluster contient {len(P[max(P.keys())])} éléments")

    # Affichage du dendrogramme si demandé
    if dendrogramme:
        plt.figure()
        sch.dendrogram(Z)
        plt.title("Dendrogramme du clustering hiérarchique")
        plt.xlabel("Indices des exemples")
        plt.ylabel("Distance entre clusters")
        plt.show()

    return fusion_results

def dist_max(group1, group2):
    """Calcule la distance maximale entre deux groupes d'exemples."""
    max_dist = -np.inf
    for i in group1:
        for j in group2:
            d = dist_euclidienne(i, j)
            if d > max_dist:
                max_dist = d
    return max_dist

def dist_min(group1, group2):
    """Calcule la distance minimale entre deux groupes d'exemples."""
    min_dist = np.inf
    for i in group1:
        for j in group2:
            d = dist_euclidienne(i, j)
            if d < min_dist:
                min_dist = d
    return min_dist

def dist_average(group1, group2):
    """Calcule la distance moyenne entre deux groupes d'exemples."""
    total_dist = 0
    count = 0
    for i in group1:
        for j in group2:
            d = dist_euclidienne(i, j)
            total_dist += d
            count += 1
    return total_dist / count

def CHA_complete(df, verbose=False, dendrogramme=False):
    """Clustering hiérarchique ascendant avec linkage complet."""
    P = initialise_CHA(df)
    fusion_results = []
    Z = []

    while len(P) > 1:
        c1, c2 = None, None
        d_min = float("inf")

        clusters = list(P.keys())
        for i, j in itertools.combinations(clusters, 2):
            d = dist_max(df.iloc[P[i]].values, df.iloc[P[j]].values)
            if d < d_min:
                d_min = d
                c1, c2 = i, j

        new_cluster = P[c1] + P[c2]
        P1 = {k: v for k, v in P.items() if k not in (c1, c2)}
        new_key = max(P.keys()) + 1
        P1[new_key] = new_cluster
        P = P1

        fusion_results.append([c1, c2, d_min, len(P[new_key])])
        Z.append([c1, c2, d_min, len(P[new_key])])

        if verbose:
            print(f"CHA_complete: Fusion des clusters {c1} et {c2} à une distance de {d_min:.16f}")

    if dendrogramme:
        plt.figure()
        sch.dendrogram(Z)
        plt.title("Dendrogramme CHA Complet")
        plt.show()

    return fusion_results

def CHA_simple(df, verbose=False, dendrogramme=False):
    """Clustering hiérarchique ascendant avec linkage simple."""
    P = initialise_CHA(df)
    fusion_results = []
    Z = []

    while len(P) > 1:
        c1, c2 = None, None
        d_min = float("inf")

        clusters = list(P.keys())
        for i, j in itertools.combinations(clusters, 2):
            d = dist_min(df.iloc[P[i]].values, df.iloc[P[j]].values)
            if d < d_min:
                d_min = d
                c1, c2 = i, j

        new_cluster = P[c1] + P[c2]
        P1 = {k: v for k, v in P.items() if k not in (c1, c2)}
        new_key = max(P.keys()) + 1
        P1[new_key] = new_cluster
        P = P1

        fusion_results.append([c1, c2, d_min, len(P[new_key])])
        Z.append([c1, c2, d_min, len(P[new_key])])

        if verbose:
            print(f"CHA_simple: Fusion des clusters {c1} et {c2} à une distance de {d_min:.16f}")

    if dendrogramme:
        plt.figure()
        sch.dendrogram(Z)
        plt.title("Dendrogramme CHA Simple")
        plt.show()

    return fusion_results

def CHA_average(df, verbose=False, dendrogramme=False):
    """Clustering hiérarchique ascendant avec linkage moyen."""
    P = initialise_CHA(df)
    fusion_results = []
    Z = []

    while len(P) > 1:
        c1, c2 = None, None
        d_min = float("inf")

        clusters = list(P.keys())
        for i, j in itertools.combinations(clusters, 2):
            d = dist_average(df.iloc[P[i]].values, df.iloc[P[j]].values)
            if d < d_min:
                d_min = d
                c1, c2 = i, j

        new_cluster = P[c1] + P[c2]
        P1 = {k: v for k, v in P.items() if k not in (c1, c2)}
        new_key = max(P.keys()) + 1
        P1[new_key] = new_cluster
        P = P1

        fusion_results.append([c1, c2, d_min, len(P[new_key])])
        Z.append([c1, c2, d_min, len(P[new_key])])

        if verbose:
            print(f"CHA_average: Fusion des clusters {c1} et {c2} à une distance de {d_min:.16f}")

    if dendrogramme:
        plt.figure()
        sch.dendrogram(Z)
        plt.title("Dendrogramme CHA Moyen")
        plt.show()

    return fusion_results

def CHA(df, linkage, verbose=False, dendrogramme=False):
    """
    Effectue le clustering hiérarchique ascendant (CHA) en utilisant différentes méthodes de linkage.

    Paramètres:
        - df : DataFrame contenant les données normalisées.
        - linkage : Chaîne de caractères spécifiant la méthode de linkage à utiliser.
                    Peut être 'centroid', 'complete', 'simple', ou 'average'.
                    Valeur par défaut : 'centroid'.
        - verbose : Booléen, affiche des messages si True.
        - dendrogramme : Booléen, affiche le dendrogramme si True.

    Retourne:
        - Une liste composée de listes contenant chacune :
            - Les 2 indices d'éléments fusionnés
            - La distance les séparant
            - La somme du nombre d'éléments des 2 éléments fusionnés
    """
    # Initialisation de la partition
    P = initialise_CHA(df)
    fusion_results = []
    Z = []

    while len(P) > 1:
        c1, c2 = None, None
        d_min = float("inf")

        clusters = list(P.keys())
        for i, j in itertools.combinations(clusters, 2):
            if linkage == 'centroid':
                d = dist_centroides(df.iloc[P[i]], df.iloc[P[j]])
            elif linkage == 'complete':
                d = dist_max(df.iloc[P[i]].values, df.iloc[P[j]].values)
            elif linkage == 'simple':
                d = dist_min(df.iloc[P[i]].values, df.iloc[P[j]].values)
            elif linkage == 'average':
                d = dist_average(df.iloc[P[i]].values, df.iloc[P[j]].values)
            else:
                raise ValueError("Méthode de linkage inconnue. Utilisez 'centroid', 'complete', 'simple', ou 'average'.")

            if d < d_min:
                d_min = d
                c1, c2 = i, j

        new_cluster = P[c1] + P[c2]
        P1 = {k: v for k, v in P.items() if k not in (c1, c2)}
        new_key = max(P.keys()) + 1
        P1[new_key] = new_cluster
        P = P1

        fusion_results.append([c1, c2, d_min, len(P[new_key])])
        Z.append([c1, c2, d_min, len(P[new_key])])

        if verbose:
            print(f"CHA ({linkage}): Fusion des clusters {c1} et {c2} à une distance de {d_min:.16f}")

    if dendrogramme:
        plt.figure()
        sch.dendrogram(Z)
        plt.title(f"Dendrogramme CHA {linkage.capitalize()}")
        plt.show()

    return fusion_results

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """

    Ens = np.array(Ens)
    
    # Calcul du centroïde
    centre = np.mean(Ens, axis=0)

    # Calcul des distances au carré de chaque point au centroïde
    distances_carre = np.sum((Ens - centre) ** 2, axis=1)

    # Inertie = somme des distances au carré
    return np.sum(distances_carre)

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """

    indices = np.random.choice(len(Ens), size=K, replace=False)
    return Ens.iloc[indices] 
    

def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """

    # Calcul des distances euclidiennes entre Exe et chaque centre
    distances = np.linalg.norm(Centres - Exe, axis=1)
    return np.argmin(distances)
    

def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    
    # Initialisation d'un dictionnaire pour l'affectation
    affectation = {i: [] for i in range(len(Centres))}
    
    # Pour chaque exemple de la base, on l'affecte au cluster du centroïde le plus proche
    for i, exemple in enumerate(Base.values):
        # Trouver l'indice du centroïde le plus proche
        indice_cluster = plus_proche(exemple, Centres)
        # Ajouter cet exemple à la liste du cluster correspondant
        affectation[indice_cluster].append(i)
    
    return affectation  
    
def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    centroides = []
    
    # Pour chaque cluster (chaque clé dans U)
    for cluster, indices in U.items():
        # Sélectionner les exemples appartenant au cluster
        cluster_data = Base.iloc[indices] if isinstance(Base, pd.DataFrame) else Base[indices]
        
        # Calculer le centroïde du cluster (moyenne des exemples)
        centroide = cluster_data.mean(axis=0)
        
        # Ajouter le centroïde à la liste
        centroides.append(centroide)
    
    # Convertir la liste des centroïdes en DataFrame
    nouveaux_centroides_df = pd.DataFrame(centroides)
    
    return nouveaux_centroides_df     

def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    inertie_totale = 0.0
    
    # Parcours des clusters dans U
    for cluster_id, indices in U.items():
        # Récupération des points du cluster à partir de Base
        points = Base.iloc[indices]  # Utilise .iloc pour accéder aux lignes par index
        
        # Calcul de l'inertie pour ce cluster avec la fonction inertie_cluster
        inertie_totale += inertie_cluster(points)
    
    return inertie_totale

def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0 (critère de convergence)
        iter_max : entier >1 (nombre d'itérations maximal)
    """
    
    # Initialisation des centroids : on choisit aléatoirement K points de la base
    indices_initials = np.random.choice(Base.shape[0], K, replace=False)
    centroids = Base.iloc[indices_initials].values  # Utilise .iloc[] pour accéder aux lignes

    # Initialisation de la matrice d'affectation
    DictAffect = {i: [] for i in range(K)}

    # Calcul de l'inertie initiale
    inertie_precedente = 1 + epsilon

    for iteration in range(iter_max):
        # 1. Assignation des points aux clusters
        DictAffect = {i: [] for i in range(K)}  # On réinitialise les clusters
        for i, point in enumerate(Base.values):
            # Calcul des distances au centroid le plus proche
            distances = np.linalg.norm(centroids - point, axis=1)
            cluster_id = np.argmin(distances)
            DictAffect[cluster_id].append(i)
        
        # 2. Mise à jour des centroids
        nouveaux_centroids = np.zeros_like(centroids)
        for i in range(K):
            # Calcul du centroïde de chaque cluster (moyenne des points)
            if len(DictAffect[i]) > 0:
                nouveaux_centroids[i] = np.mean(Base.iloc[DictAffect[i]].values, axis=0)

        # 3. Calcul de l'inertie pour cette itération
        inertie_actuelle = 0
        for i in range(K):
            cluster_points = Base.iloc[DictAffect[i]].values
            inertie_actuelle += inertie_cluster(cluster_points)

        # 4. Calcul de la différence avec l'inertie précédente
        difference_inertie = abs(inertie_actuelle - inertie_precedente)  # Utiliser la valeur absolue

        # Affichage de l'inertie et de la différence
        print(f"Iteration {iteration + 1} Inertie : {inertie_actuelle:.4f} Difference: {difference_inertie:.4f}")

        # Critère de convergence : vérifier si l'inertie a suffisamment diminué
        if difference_inertie < epsilon:
            break

        # Mettre à jour les centroids et l'inertie précédente pour la prochaine itération
        centroids = nouveaux_centroids
        inertie_precedente = inertie_actuelle

    return centroids, DictAffect

def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """

    # Nombre de clusters K
    K = len(Centres)

    # On récupère les couleurs de la colormap 'tab20' (maximum 20 couleurs)
    couleurs = cm.tab20(np.linspace(0, 1, 20))

    # Création de la figure et des axes pour l'affichage
    plt.figure(figsize=(8, 6))

    # Pour chaque cluster, on affiche les points avec la couleur correspondante
    for cluster_id in range(K):
        # Récupérer les indices des points appartenant à ce cluster
        indices = Affect[cluster_id]
        # Extraire les points de Base correspondants
        points_cluster = Base.iloc[indices]
        
        # Tracer les points du cluster avec la couleur correspondante
        plt.scatter(points_cluster.iloc[:, 0], points_cluster.iloc[:, 1], 
                    color=couleurs[cluster_id], label=f'Cluster {cluster_id+1}')
        
        # Afficher le centre du cluster en rouge et en plus gros
        plt.scatter(Centres[cluster_id, 0], Centres[cluster_id, 1], 
                    color='red', marker='x', s=100, label=f'Centre {cluster_id+1}')

    # Ajouter des légendes, titre et axes
    plt.title('Résultats de l\'algorithme K-means')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    
    # Affichage du graphique
    plt.show()   
# ------------------------ 
