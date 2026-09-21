# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
import scipy.stats as stats
from typing import List, Tuple
import utils

def sufficient_statistics(data: np.ndarray, dico: np.ndarray, x: int, y: int, z: List[int]) -> Tuple[float, int]:
    """
    Calcule la statistique du chi-deux conditionnelle et les degrés de liberté
    
    Paramètres:
        data: tableau de données
        dico: tableau de dictionnaires d'encodage
        x: index de la variable X
        y: index de la variable Y  
        z: liste d'index des variables conditionnelles Z
    
    Retourne:
        (valeur du chi-deux, degrés de liberté)
    """
    # Obtenir les tables de contingence
    contingency_tables = utils.create_contingency_table(data, dico, x, y, z)
    
    chi_square = 0.0
    valid_z_count = 0  # Compter le nombre de valeurs z valides
    
    for Nz, T_xy_z in contingency_tables:
        if Nz == 0:
            continue
            
        valid_z_count += 1
        
        # Calculer les sommes marginales
        N_xz = np.sum(T_xy_z, axis=1)  # Somme sur y pour obtenir N_xz
        N_yz = np.sum(T_xy_z, axis=0)  # Somme sur x pour obtenir N_yz
        
        # Parcourir toutes les combinaisons x,y
        for i in range(T_xy_z.shape[0]):  # Valeurs de x
            for j in range(T_xy_z.shape[1]):  # Valeurs de y
                N_xyz = T_xy_z[i, j]
                # Calculer la fréquence attendue
                expected = (N_xz[i] * N_yz[j]) / Nz
                
                # Si la fréquence attendue n'est pas nulle, calculer le terme chi-deux
                if expected > 0:
                    # Formule correcte du chi-deux
                    chi_square += ((N_xyz - expected) ** 2) / expected
    
    # Calculer les degrés de liberté
    if valid_z_count == 0:
        dof = 0
    else:
        # Obtenir le nombre de valeurs possibles pour X et Y
        x_cardinality = len(dico[x])
        y_cardinality = len(dico[y])
        dof = (x_cardinality - 1) * (y_cardinality - 1) * valid_z_count
    
    return chi_square, dof

def indep_score(data: np.ndarray, dico: np.ndarray, x: int, y: int, z: List[int]) -> Tuple[float, int]:
    """
    Calcule la p-value du test d'indépendance et les degrés de liberté
    
    Paramètres:
        data: tableau de données
        dico: tableau de dictionnaires d'encodage
        x: index de la variable X
        y: index de la variable Y
        z: liste d'index des variables conditionnelles Z
    
    Retourne:
        (p-value, degrés de liberté)
    """
    # Vérifier si le test est valide (taille d'échantillon suffisante)
    n_records = len(data[0])  # Nombre d'enregistrements de données
    
    # Calculer la taille minimale requise
    x_cardinality = len(dico[x])
    y_cardinality = len(dico[y])
    z_cardinality = 1
    for zi in z:
        z_cardinality *= len(dico[zi])
    
    d_min = 5 * x_cardinality * y_cardinality * z_cardinality
    
    if n_records < d_min:
        return (-1, 0)  # Test non valide, retourner -1
    
    # Calculer la statistique du chi-deux et les degrés de liberté
    chi_sq, dof = sufficient_statistics(data, dico, x, y, z)
    
    if dof == 0:
        return (1.0, 0)  # Degrés de liberté nuls, indépendance complète
    
    # Calculer la p-value
    p_value = stats.chi2.sf(chi_sq, dof)
    
    return (p_value, dof)

def best_candidate(data: np.ndarray, dico: np.ndarray, x: int, z: List[int], alpha: float) -> List[int]:
    """
    Trouve la variable la plus dépendante de X conditionnellement à Z
    
    Paramètres:
        data: tableau de données
        dico: tableau de dictionnaires d'encodage
        x: index de la variable cible
        z: liste des parents actuels
        alpha: niveau de signification
    
    Retourne:
        Liste d'index de la variable la plus corrélée (vide si indépendante)
    """
    n_variables = data.shape[0]
    best_p_value = 1.0  # Initialiser la meilleure p-value
    best_candidate = -1  # Initialiser le meilleur candidat
    
    # Parcourir toutes les variables à gauche de x (selon l'énoncé)
    for y in range(x):  # Ne considérer que les variables à gauche de x
        if y in z:  # Sauter les variables déjà parents
            continue
            
        p_value, _ = indep_score(data, dico, x, y, z)
        
        # Si le test n'est pas valide, passer
        if p_value == -1:
            continue
            
        # Mettre à jour le meilleur candidat
        if p_value < best_p_value:
            best_p_value = p_value
            best_candidate = y
    
    # Vérifier l'indépendance
    if best_p_value > alpha or best_candidate == -1:
        return []  # Indépendante, retourner liste vide
    else:
        return [best_candidate]  # Dépendante, retourner le candidat

def create_parents(data: np.ndarray, dico: np.ndarray, x: int, alpha: float) -> List[int]:
    """
    Crée l'ensemble des parents pour une variable x
    
    Paramètres:
        data: tableau de données
        dico: tableau de dictionnaires d'encodage
        x: index de la variable cible
        alpha: niveau de signification
    
    Retourne:
        Liste d'index des parents
    """
    parents = []  # Initialiser l'ensemble des parents
    
    while True:
        # Trouver le meilleur candidat
        candidate = best_candidate(data, dico, x, parents, alpha)
        
        # S'il n'y a plus de variables dépendantes, arrêter
        if not candidate:
            break
            
        # Ajouter le meilleur candidat à l'ensemble des parents
        parents.extend(candidate)
    
    return parents

def learn_BN_structure(data: np.ndarray, dico: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apprend la structure complète du réseau bayésien

    Paramètres:
        data: tableau de données
        dico: tableau de dictionnaires d'encodage
        alpha: niveau de signification

    Retourne:
        Tableau numpy des listes de parents pour chaque nœud
    """
    n_variables = data.shape[0]
    bn_structure = []

    # Apprendre les parents pour chaque variable
    for x in range(n_variables):
        parents = create_parents(data, dico, x, alpha)
        bn_structure.append(parents)

    # Convertir en tableau numpy pour la compatibilité
    return np.array(bn_structure, dtype=object)