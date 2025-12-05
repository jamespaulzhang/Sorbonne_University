# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi
import pyAgrum as gum

def bernoulli(p: float) -> int:
    """
    Simule une variable aléatoire de Bernoulli.
    
    Args:
        p: probabilité de succès (valeur dans [0,1])
    
    Returns:
        1 avec probabilité p, 0 avec probabilité 1-p
    """
    if p < 0 or p > 1:
        raise ValueError("p doit être dans l'intervalle [0,1]")
    
    # Génère un nombre aléatoire uniforme dans [0,1]
    # et retourne 1 si ce nombre est inférieur à p
    return 1 if random.random() < p else 0

def binomiale(n: int, p: float) -> int:
    """
    Simule une variable aléatoire suivant une loi binomiale B(n,p).
    
    Args:
        n: nombre d'épreuves de Bernoulli
        p: probabilité de succès pour chaque épreuve
    
    Returns:
        Nombre de succès sur n épreuves indépendantes
    """
    if n < 0:
        raise ValueError("n doit être un entier positif")
    if p < 0 or p > 1:
        raise ValueError("p doit être dans l'intervalle [0,1]")
    
    # Somme de n variables de Bernoulli indépendantes
    return sum(bernoulli(p) for _ in range(n))

def galton(l: int, n: int, p: float = 0.5) -> np.ndarray:
    """
    Simule l expériences de la planche de Galton (loi binomiale B(n,p)).
    
    Args:
        l: nombre d'expériences (billes)
        n: nombre de niveaux (hauteur de la planche)
        p: probabilité d'aller à droite à chaque niveau (défaut: 0.5)
    
    Returns:
        Tableau de l valeurs suivant la loi binomiale B(n,p)
    """
    if l <= 0:
        raise ValueError("l doit être un entier positif")
    if n <= 0:
        raise ValueError("n doit être un entier positif")
    if p < 0 or p > 1:
        raise ValueError("p doit être dans l'intervalle [0,1]")
    
    # Génère l variables aléatoires binomiales B(n,p)
    return np.array([binomiale(n, p) for _ in range(l)])

def histo_galton(l: int, n: int, p: float = 0.5):
    """
    Trace l'histogramme de la répartition des billes dans la planche de Galton.
    Utilise le nombre de valeurs différentes dans les données comme nombre de bins.
    
    Args:
        l: nombre de billes
        n: nombre de niveaux (hauteur de la planche)
        p: probabilité de déviation à droite (0.5 par défaut)
    """
    # Génère les données expérimentales de la planche de Galton
    resultats = galton(l, n, p)
    
    # Calcule le nombre de valeurs distinctes dans les données pour déterminer le nombre de bins
    valeurs_uniques = np.unique(resultats)
    nb_bins = len(valeurs_uniques)
    
    print(f"Nombre de valeurs différentes dans les données: {nb_bins}")
    print(f"Valeurs observées: {valeurs_uniques}")
    
    # Crée la figure pour l'histogramme
    plt.figure(figsize=(12, 8))
    
    # Trace l'histogramme en utilisant le nombre de bins calculé
    plt.hist(resultats, bins=nb_bins, density=False, edgecolor='black', linewidth=1.2)
    
    # Ajoute le titre et les labels des axes
    plt.title(f'Répartition des {l} billes dans la planche de Galton\n'
              f'n={n}, p={p} - {nb_bins} cases utilisées', fontsize=14)
    plt.xlabel('Numéro de la case', fontsize=12)
    plt.ylabel('Densité de probabilité', fontsize=12)
    
    # Ajuste la mise en page et affiche le graphique
    plt.tight_layout()
    plt.show()

def normale(k: int, sigma: float) -> np.ndarray:
    """
    Génère les valeurs de la fonction de densité de la loi normale N(0,σ²).
    
    Args:
        k: nombre de points (doit être impair pour la symétrie)
        sigma: écart-type de la distribution normale
    
    Returns:
        Array numpy des k valeurs y_i de la fonction de densité
    
    Raises:
        ValueError: si k est pair
    """
    # Vérifie que k est impair
    if k % 2 == 0:
        raise ValueError(f"k doit être impair, mais k={k} est pair")
    
    # Crée les points x équi-espacés de -2σ à 2σ
    x = np.linspace(-2 * sigma, 2 * sigma, k)
    
    # Calcule les valeurs y de la fonction de densité N(0,σ²)
    y = (1 / (sigma * np.sqrt(2 * pi))) * np.exp(-0.5 * (x / sigma) ** 2)
    
    return y

def proba_affine(k: int, slope: float) -> np.ndarray:
    """
    Génère une distribution de probabilité affine.
    
    Args:
        k: nombre de points (doit être impair)
        slope: pente de la distribution affine
    
    Returns:
        Array numpy des k valeurs y_i représentant la distribution de probabilité
    
    Raises:
        ValueError: si k est pair ou si la pente est trop élevée
    """
    # Vérifie que k est impair
    if k % 2 == 0:
        raise ValueError(f"k doit être impair, mais k={k} est pair")
    
    # Calcule la pente maximale possible pour assurer que toutes les probabilités sont positives
    pente_max = 1 / (k * ((k - 1) / 2))
    
    if abs(slope) > pente_max:
        raise ValueError(f"La pente {slope} est trop élevée. Pente maximale possible: {pente_max:.6f}")
    
    # Calcule les valeurs y_i selon la formule donnée
    y = np.array([1/k + (i - (k-1)/2) * slope for i in range(k)])
    
    # Vérification que toutes les probabilités sont positives
    if np.any(y < 0):
        raise ValueError("Certaines probabilités sont négatives. Réduisez la pente.")
    
    return y

def Pxy(PA: np.ndarray, PB: np.ndarray) -> np.ndarray:
    """
    Calcule la distribution jointe P(A,B) pour deux variables aléatoires indépendantes.
    
    Args:
        PA: distribution de probabilité de A (1D array)
        PB: distribution de probabilité de B (1D array)
    
    Returns:
        Tableau 2D numpy représentant la distribution jointe P(A,B)
    """

    # Calcule la distribution jointe
    P_jointe = np.outer(PA, PB)
    
    return P_jointe

def calcYZ(P_XYZT: np.ndarray) -> np.ndarray:
    """
    Calcule la distribution marginale P(Y,Z) à partir de P(X,Y,Z,T).
    
    Args:
        P_XYZT: tableau 4D représentant P(X,Y,Z,T)
    
    Returns:
        Tableau 2D représentant P(Y,Z)
    """
    # Vérification de la forme du tableau
    if P_XYZT.shape != (2, 2, 2, 2):
        raise ValueError("P_XYZT doit être un tableau 2×2×2×2")
    
    # Initialisation du tableau P(Y,Z)
    P_YZ = np.zeros((2, 2))
    
    # Calcul de P(Y,Z) = Σ_X Σ_T P(X,Y,Z,T)
    for y in range(2):      # pour chaque valeur de Y
        for z in range(2):  # pour chaque valeur de Z
            # Somme sur X et T
            total = 0
            for x in range(2):
                for t in range(2):
                    total += P_XYZT[x][y][z][t]
            P_YZ[y][z] = total
    
    return P_YZ

def calcXTcondYZ(P_XYZT: np.ndarray) -> np.ndarray:
    """
    Calcule la distribution conditionnelle P(X,T|Y,Z) à partir de P(X,Y,Z,T).
    
    Args:
        P_XYZT: tableau 4D représentant P(X,Y,Z,T)
    
    Returns:
        Tableau 4D représentant P(X,T|Y,Z)
    """
    # Vérification de la forme du tableau
    if P_XYZT.shape != (2, 2, 2, 2):
        raise ValueError("P_XYZT doit être un tableau 2×2×2×2")
    
    # Calcul de P(Y,Z) en utilisant la fonction précédente
    P_YZ = calcYZ(P_XYZT)
    
    # Initialisation du tableau P(X,T|Y,Z)
    P_XTcondYZ = np.zeros((2, 2, 2, 2))
    
    # Calcul de P(X,T|Y,Z) = P(X,Y,Z,T) / P(Y,Z)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    # Éviter la division par zéro
                    if P_YZ[y][z] > 0:
                        P_XTcondYZ[x][y][z][t] = P_XYZT[x][y][z][t] / P_YZ[y][z]
                    else:
                        P_XTcondYZ[x][y][z][t] = 0.0
    
    return P_XTcondYZ

def calcX_etTcondYZ(P_XYZT: np.ndarray):
    """
    Calcule les distributions conditionnelles P(X|Y,Z) et P(T|Y,Z) à partir de P(X,Y,Z,T).
    
    Args:
        P_XYZT: tableau 4D représentant P(X,Y,Z,T)
    
    Returns:
        Tuple (P_XcondYZ, P_TcondYZ) où:
        - P_XcondYZ: tableau 3D représentant P(X|Y,Z)
        - P_TcondYZ: tableau 3D représentant P(T|Y,Z)
    """
    # Vérification de la forme du tableau
    if P_XYZT.shape != (2, 2, 2, 2):
        raise ValueError("P_XYZT doit être un tableau 2×2×2×2")
    
    # Calcul de P(Y,Z)
    P_YZ = calcYZ(P_XYZT)
    
    # Initialisation des tableaux
    P_XcondYZ = np.zeros((2, 2, 2))  # Dimensions: X, Y, Z
    P_TcondYZ = np.zeros((2, 2, 2))  # Dimensions: T, Y, Z
    
    # Calcul de P(X|Y,Z) = Σ_T P(X,T,Y,Z) / P(Y,Z)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                if P_YZ[y][z] > 0:
                    # Somme sur T
                    somme_sur_T = 0
                    for t in range(2):
                        somme_sur_T += P_XYZT[x][y][z][t]
                    P_XcondYZ[x][y][z] = somme_sur_T / P_YZ[y][z]
                else:
                    P_XcondYZ[x][y][z] = 0.0
    
    # Calcul de P(T|Y,Z) = Σ_X P(X,T,Y,Z) / P(Y,Z)
    for t in range(2):
        for y in range(2):
            for z in range(2):
                if P_YZ[y][z] > 0:
                    # Somme sur X
                    somme_sur_X = 0
                    for x in range(2):
                        somme_sur_X += P_XYZT[x][y][z][t]
                    P_TcondYZ[t][y][z] = somme_sur_X / P_YZ[y][z]
                else:
                    P_TcondYZ[t][y][z] = 0.0
    
    return P_XcondYZ, P_TcondYZ

def testXTindepCondYZ(P_XYZT: np.ndarray, epsilon: float = 1e-10) -> bool:
    """
    Teste si X et T sont indépendants conditionnellement à (Y,Z) dans la distribution P_XYZT.
    
    Args:
        P_XYZT: tableau 4D représentant P(X,Y,Z,T)
        epsilon: tolérance numérique pour la comparaison
    
    Returns:
        True si X et T sont conditionnellement indépendants donné (Y,Z), False sinon
    """
    # Vérification de la forme du tableau
    if P_XYZT.shape != (2, 2, 2, 2):
        raise ValueError("P_XYZT doit être un tableau 2×2×2×2")
    
    # Calcul des distributions nécessaires
    P_YZ = calcYZ(P_XYZT)
    P_XTcondYZ = calcXTcondYZ(P_XYZT)
    P_XcondYZ, P_TcondYZ = calcX_etTcondYZ(P_XYZT)
    
    # Vérification de l'indépendance conditionnelle
    # P(X,T|Y,Z) doit être égal à P(X|Y,Z) × P(T|Y,Z) pour tous X,T,Y,Z
    
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for t in range(2):
                    # Calcul du produit P(X|Y,Z) × P(T|Y,Z)
                    produit = P_XcondYZ[x][y][z] * P_TcondYZ[t][y][z]
                    
                    # Valeur réelle de P(X,T|Y,Z)
                    valeur_reelle = P_XTcondYZ[x][y][z][t]
                    
                    # Comparaison avec tolérance
                    if not np.isclose(produit, valeur_reelle, atol=epsilon):
                        print(f"Échec pour X={x}, Y={y}, Z={z}, T={t}:")
                        print(f"  P(X,T|Y,Z) = {valeur_reelle:.10f}")
                        print(f"  P(X|Y,Z) × P(T|Y,Z) = {P_XcondYZ[x][y][z]:.10f} × {P_TcondYZ[t][y][z]:.10f} = {produit:.10f}")
                        print(f"  Différence: {abs(valeur_reelle - produit):.2e}")
                        return False
    
    return True

def testXindepYZ(P_XYZT: np.ndarray, epsilon: float = 1e-10) -> bool:
    """
    Teste si X et (Y,Z) sont indépendants dans la distribution P_XYZT.
    
    Args:
        P_XYZT: tableau 4D représentant P(X,Y,Z,T)
        epsilon: tolérance numérique pour la comparaison
    
    Returns:
        True si X et (Y,Z) sont indépendants, False sinon
    """
    # Vérification de la forme du tableau
    if P_XYZT.shape != (2, 2, 2, 2):
        raise ValueError("P_XYZT doit être un tableau 2×2×2×2")
    
    # 1. Calcul de P(X,Y,Z) = Σ_T P(X,Y,Z,T)
    P_XYZ = np.sum(P_XYZT, axis=3)
    
    # 2. Calcul de P(X) = Σ_Y Σ_Z P(X,Y,Z)
    P_X = np.sum(P_XYZ, axis=(1, 2))
    
    # Calcul de P(Y,Z) = Σ_X P(X,Y,Z)
    P_YZ = np.sum(P_XYZ, axis=0)
    
    # 3. Vérification de l'indépendance: P(X,Y,Z) = P(X) × P(Y,Z)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                # Calcul du produit P(X) × P(Y,Z)
                produit = P_X[x] * P_YZ[y][z]
                
                # Valeur réelle de P(X,Y,Z)
                valeur_reelle = P_XYZ[x][y][z]
                
                # Comparaison avec tolérance
                if not np.isclose(produit, valeur_reelle, atol=epsilon):
                    return False
                
    return True

def conditional_indep(join_potential: gum.Potential,
                     var_X: str,
                     var_Y: str,
                     conditioning_vars: list,
                     epsilon: float = 1e-10) -> bool:
    """
    Teste si X est indépendant de Y conditionnellement à Z dans la distribution jointe.

    Args:
        join_potential: Potential représentant la distribution jointe P(X,Y,Z)
        var_X: nom de la variable X
        var_Y: nom de la variable Y
        conditioning_vars: liste des noms des variables de conditionnement Z
        epsilon: tolérance numérique pour la comparaison

    Returns:
        True si X ⊥ Y | Z, False sinon
    """
    if len(conditioning_vars) == 0:
        # Cas sans conditionnement: P(X,Y) == P(X)P(Y)
        pXY = join_potential.sumIn([var_X, var_Y])
        pX = pXY.sumOut([var_Y])
        pY = pXY.sumOut([var_X])
        return (pXY - (pX * pY)).abs().max() < epsilon
    else:
        # Cas avec conditionnement: P(X,Y|Z) == P(X|Z)P(Y|Z)
        vars_to_keep = [var_X, var_Y] + conditioning_vars
        pXYZ = join_potential.sumIn(vars_to_keep)
        pZ = pXYZ.sumOut([var_X, var_Y])
        pXY_Z = pXYZ / pZ
        pX_Z = pXY_Z.sumOut([var_Y])
        pY_Z = pXY_Z.sumOut([var_X])
        return (pXY_Z - (pX_Z * pY_Z)).abs().max() < epsilon
    
def compact_conditional_proba(joint_proba: gum.Potential, Xin: str, epsilon: float = 1e-10) -> gum.Potential:
    """
    Retourne P(Xin | K_compact) où K_compact ⊆ toutes les autres variables
    et toutes les variables conditionnellement indépendantes de Xin sont retirées.
    """
    # 1) Liste initiale K = toutes les variables sauf Xin
    all_vars = list(joint_proba.var_names)
    if Xin not in all_vars:
        raise ValueError(f"{Xin} n'est pas dans joint_proba.var_names")
    K = [v for v in all_vars if v != Xin]

    # 2) Supprimer les variables X de K qui sont indépendantes de Xin conditionnellement à K\{X}
    changed = True
    while changed:
        changed = False
        for X in list(K):  # on travaille sur une copie
            given = [v for v in K if v != X]  # K \ {X}
            if conditional_indep(joint_proba, X, Xin, given, epsilon):
                K.remove(X)
                changed = True  # on continue jusqu'à stabilisation

    # 3) Construire P(Xin | K)
    vars_to_keep = [Xin] + K  # Xin en premier
    numerator = joint_proba.sumIn(vars_to_keep)  # P(Xin, K)
    denominator = numerator.sumOut([Xin])       # P(K)
    conditional = numerator / denominator       # P(Xin | K)

    # 4) Remettre Xin en première position (déjà fait ci-dessus)
    conditional = conditional.putFirst(Xin)

    return conditional

def create_bayesian_network(joint_proba: gum.Potential, epsilon: float = 1e-10) -> list:
    """
    Transforme une distribution de probabilité jointe en un réseau bayésien compact
    en utilisant l'ordre inverse des variables.
    
    Args:
        joint_proba: Potential représentant la distribution jointe P(X0,...,Xn)
        epsilon: tolérance numérique pour les tests d'indépendance conditionnelle
    
    Returns:
        Liste des Potentials [P(Xn|Kn), ..., P(X0|K0)] représentant le réseau bayésien
        dans l'ordre inverse de traitement des variables
    """
    # Création d'une copie de la distribution jointe pour ne pas modifier l'originale
    try:
        P = joint_proba.clone()
    except AttributeError:
        P = gum.Potential(joint_proba)

    # Liste qui contiendra les distributions conditionnelles du réseau bayésien
    bn_list = []
    
    # Obtention de la liste des variables et inversion de l'ordre
    var_names = list(P.var_names)
    reverse_order = list(reversed(var_names))

    # Parcours des variables dans l'ordre inverse (de Xn à X0)
    for Xi in reverse_order:
        # Calcul de la distribution conditionnelle compacte P(Xi|Ki)
        # où Ki est le sous-ensemble minimal de variables nécessaires
        Q = compact_conditional_proba(P, Xi, epsilon)
        
        # Ajout de la distribution conditionnelle à la liste du réseau bayésien
        bn_list.append(Q)
        
        # Marginalisation : suppression de Xi de la distribution jointe
        # pour la prochaine itération (principe de décomposition)
        P = P.sumOut([Xi])
    
    # Retourne la liste des distributions conditionnelles dans l'ordre inverse
    # (dernière variable traitée en premier)
    return bn_list

def calcNbParams(joint_proba: gum.Potential, epsilon: float = 1e-10) -> tuple[int, int]:
    """
    Calcule le nombre de paramètres de la loi jointe et du réseau bayésien compact.
    
    Args:
        joint_proba: Potential représentant la distribution jointe P(X0,...,Xn)
        epsilon: tolérance numérique pour l'indépendance conditionnelle
    
    Returns:
        tuple (taille_jointe, taille_rb) :
        - taille_jointe : nombre de paramètres de la loi jointe
        - taille_rb : somme des paramètres des conditionnelles du réseau bayésien
    """
    # 1) Taille de la loi jointe
    taille_jointe = joint_proba.domainSize()
    
    # 2) Création du réseau bayésien en utilisant le même epsilon
    bn_list = create_bayesian_network(joint_proba, epsilon)
    
    # 3) Taille du réseau bayésien = somme des tailles de chaque Potential
    taille_rb = sum(Q.domainSize() for Q in bn_list)
    
    return taille_jointe, taille_rb