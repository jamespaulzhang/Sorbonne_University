# Yuxiang ZHANG 21202829
# Sam ASLO 21210657

import pandas as pd
import numpy as np
from utils import *
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Question 1.1 : calcul de la probabilité a priori
def getPrior(df):
    """
    Calcule les probabilités a priori pour la classe cible (target) dans un DataFrame.

    Cette fonction calcule la probabilité d'appartenance à la classe 1 (p) et les intervalles de confiance 
    à 95% pour cette probabilité (min5pourcent et max5pourcent).

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données, avec une colonne 'target' représentant la classe cible.
    
    Returns
    -------
    dict
        Un dictionnaire contenant la probabilité a priori de la classe 1 ('estimation'), 
        ainsi que les bornes inférieure et supérieure de l'intervalle de confiance à 95% 
        ('min5pourcent', 'max5pourcent').
    """
    n = len(df)
    
    count_class_1 = df['target'].sum()
    
    p = count_class_1 / n
    
    margin = 1.96 * np.sqrt(p * (1 - p) / n)
    min_ci = p - margin
    max_ci = p + margin
    
    return {
        'estimation': p,
        'min5pourcent': min_ci,
        'max5pourcent': max_ci
    }

# Question 1.2 : programmation orientée objet dans la hiérarchie des Classifier
# Question 1.2.a
class APrioriClassifier(AbstractClassifier):
    """
    Classe de classifieur a priori qui estime la classe majoritaire en se basant uniquement sur la probabilité a priori.

    Cette classe suppose que la classe majoritaire (la classe ayant la probabilité a priori la plus élevée) 
    est la prédiction. Le classificateur se base sur l'estimation de la probabilité de chaque classe dans 
    le DataFrame d'apprentissage et prédit la classe majoritaire.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Le DataFrame d'apprentissage utilisé pour estimer les probabilités a priori.
    """
    
    def __init__(self, train_df):
        """
        Initialise le classificateur a priori en calculant la probabilité a priori de la classe cible (target).
        
        Parameters
        ----------
        train_df : pandas.DataFrame
            Le DataFrame d'apprentissage utilisé pour obtenir la probabilité a priori de la classe cible.
        """
        super().__init__()
        self.train_df = train_df
        prior = getPrior(train_df)
        self.prior_prob = prior['estimation']
        self.majority_class = 1 if self.prior_prob > 0.5 else 0
        
    def estimClass(self, attrs):
        """
        Estime la classe d'un individu en fonction de la classe majoritaire.

        Parameters
        ----------
        attrs : dict
            Un dictionnaire représentant un individu avec ses attributs.

        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        return self.majority_class

# Question 1.2.b : évaluation de classifieurs
    def statsOnDF(self, df):
        """
        Calcule les statistiques de performance (VP, VN, FP, FN, Précision, Rappel) sur un DataFrame donné.

        Parameters
        ----------
        df : pandas.DataFrame
            Le DataFrame contenant les données et les véritables classes.

        Returns
        -------
        dict
            Un dictionnaire contenant les statistiques de performance : VP, VN, FP, FN, Précision, Rappel.
        """
        VP = VN = FP = FN = 0

        for i in range(len(df)):
            dic = getNthDict(df, i)
            predicted_class = self.estimClass(dic)
            actual_class = dic['target']
            
            if predicted_class == 1 and actual_class == 1:
                VP += 1
            elif predicted_class == 0 and actual_class == 0:
                VN += 1
            elif predicted_class == 1 and actual_class == 0:
                FP += 1
            elif predicted_class == 0 and actual_class == 1:
                FN += 1
        
        precision = VP / (VP + FP) if (VP + FP) > 0 else 0
        recall = VP / (VP + FN) if (VP + FN) > 0 else 0

        return {
            'VP': VP,
            'VN': VN,
            'FP': FP,
            'FN': FN,
            'Précision': precision,
            'Rappel': recall
        }

# Question 2.1 : probabilités conditionelles
# Question 2.1.a
def P2D_l(df, attr):
    """
    Calcule les probabilités conditionnelles P(attr=a | target=t) pour un attribut donné.

    Cette fonction calcule la probabilité conditionnelle de l'attribut `attr` pour chaque valeur de la cible `target`.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr : str
        Le nom de l'attribut dont les probabilités conditionnelles doivent être calculées.

    Returns
    -------
    dict
        Un dictionnaire contenant les probabilités conditionnelles P(attr=a | target=t).
    """
    prob = {}
    
    target_values = df['target'].unique()
    attr_values = df[attr].unique()
    
    for t in target_values:
        prob[t] = {}
        target_t_df = df[df['target'] == t]
        total_t = len(target_t_df)
        
        for a in attr_values:
            count_a_given_t = len(target_t_df[target_t_df[attr] == a])
            prob[t][a] = count_a_given_t / total_t
    
    return prob

# Question 2.1.b
def P2D_p(df, attr):
    """
    Calcule les probabilités conditionnelles P(target=t | attr=a) pour un attribut donné.

    Cette fonction calcule la probabilité conditionnelle de la cible `target` pour chaque valeur de l'attribut `attr`.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr : str
        Le nom de l'attribut dont les probabilités conditionnelles doivent être calculées.

    Returns
    -------
    dict
        Un dictionnaire contenant les probabilités conditionnelles P(target=t | attr=a).
    """
    prob = {}
    
    target_values = df['target'].unique()
    attr_values = df[attr].unique()
    
    for a in attr_values:
        prob[a] = {}
        attr_a_df = df[df[attr] == a]
        total_a = len(attr_a_df)
        
        for t in target_values:
            count_t_given_a = len(attr_a_df[attr_a_df['target'] == t])
            prob[a][t] = count_t_given_a / total_a
    
    return prob

# Question 2.2 : classifieurs 2D par maximum de vraisemblance
class ML2DClassifier(APrioriClassifier):
    """
    Classifieur basé sur la méthode du maximum a posteriori (MAP) pour un attribut spécifique.

    Cette classe utilise les probabilités conditionnelles P(attr=a | target=t) pour estimer la classe 
    d'un individu en utilisant la règle de Bayes. Elle évalue la probabilité de chaque classe (0 ou 1) pour un 
    attribut donné, puis prédit la classe ayant la probabilité la plus élevée.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Le DataFrame d'apprentissage contenant les données utilisées pour calculer les probabilités.
    attr : str
        Le nom de l'attribut utilisé pour effectuer la classification (par exemple, un attribut comme "age" ou "salary").
    """
    
    def __init__(self, train_df, attr):
        """
        Initialise le classificateur ML2D en calculant les probabilités conditionnelles pour l'attribut spécifié.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Le DataFrame d'apprentissage.
        attr : str
            Le nom de l'attribut utilisé pour la classification.
        """
        super().__init__(train_df)
        self.attr = attr
        self.p2d_l = P2D_l(train_df, attr)
        self.prior = getPrior(train_df)
        self.p_attr = train_df[attr].value_counts(normalize=True).to_dict()

    def estimClass(self, attrs):
        """
        Estime la classe (0 ou 1) pour un individu donné en fonction de la valeur de l'attribut spécifié.
        
        Parameters
        ----------
        attrs : dict
            Dictionnaire contenant les attributs et leurs valeurs pour l'individu à classifier.
        
        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        attr_value = attrs[self.attr]
    
        p_attr_given_target = {t: self.p2d_l[t].get(attr_value, 0) for t in [0, 1]}
    
        p_target_0_given_attr = p_attr_given_target[0]
        p_target_1_given_attr = p_attr_given_target[1]

        # Retourne la classe avec la probabilité la plus élevée.
        if p_target_0_given_attr > p_target_1_given_attr:
            return 0
        elif p_target_1_given_attr > p_target_0_given_attr:
            return 1
        else:
            return 0

# Question 2.3 : classifieurs 2D par maximum a posteriori
class MAP2DClassifier(APrioriClassifier):
    """
    Classifieur basé sur la méthode du maximum a posteriori (MAP) pour un attribut spécifique.

    Cette classe utilise les probabilités conditionnelles P(target=t | attr=a) pour estimer la classe 
    d'un individu en utilisant la règle de Bayes avec les probabilités calculées par la fonction P2D_p.
    
    La différence avec `ML2DClassifier` est que ce modèle utilise les probabilités conditionnelles de 
    P(target=t | attr=a) pour calculer la probabilité de chaque classe, plutôt que de calculer P(attr=a | target=t).
    
    Parameters
    ----------
    train_df : pandas.DataFrame
        Le DataFrame d'apprentissage utilisé pour calculer les probabilités.
    attr : str
        Le nom de l'attribut utilisé pour effectuer la classification.
    """
    
    def __init__(self, train_df, attr):
        """
        Initialise le classificateur MAP2D en calculant les probabilités conditionnelles pour l'attribut spécifié 
        en utilisant la fonction P2D_p pour les probabilités P(target=t | attr=a).
        
        Parameters
        ----------
        train_df : pandas.DataFrame
            Le DataFrame d'apprentissage.
        attr : str
            Le nom de l'attribut utilisé pour la classification.
        """
        super().__init__(train_df)
        self.attr = attr
        self.p2d_p = P2D_p(train_df, attr)  # Utilise les probabilités conditionnelles P(target=t | attr=a)
        self.prior = getPrior(train_df)
        self.p_attr = train_df[attr].value_counts(normalize=True).to_dict()
        
    def estimClass(self, attrs):
        """
        Estime la classe (0 ou 1) pour un individu donné en fonction de la valeur de l'attribut spécifié.
        
        Parameters
        ----------
        attrs : dict
            Dictionnaire contenant les attributs et leurs valeurs pour l'individu à classifier.
        
        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        attr_value = attrs[self.attr]
        
        p_target_given_attr = {t: self.p2d_p.get(attr_value, {}).get(t, 0) for t in [0, 1]}

        p_target_0_given_attr = p_target_given_attr[0]
        p_target_1_given_attr = p_target_given_attr[1]
        
        # Retourne la classe avec la probabilité la plus élevée.
        if p_target_0_given_attr > p_target_1_given_attr:
            return 0
        elif p_target_1_given_attr > p_target_0_given_attr:
            return 1
        else:
            return 0
            
#####
# Question 2.4 : comparaison
#####
# Nous préférons le classificateur MAP2DClassifier car il offre un bon compromis entre précision et rappel, avec des résultats solides
# aussi bien en apprentissage qu'en validation. Par exemple, en validation, il atteint une précision de 0.857 et un rappel de 0.826, ce qui
# démontre qu'il est capable de bien prédire les deux classes, tout en maintenant un équilibre entre les faux positifs et les faux négatifs.

# Le classificateur ML2DClassifier présente aussi de bons résultats, notamment avec une précision élevée (0.889 en validation), mais son rappel
# (0.818) est légèrement inférieur à celui de MAP2DClassifier. Cela montre qu'il est plus performant en termes de précision, mais il pourrait manquer
# quelques cas positifs par rapport à MAP2DClassifier, ce qui peut être un inconvénient pour des applications où il est important de capturer 
# tous les cas de la classe positive.

# Enfin, le classificateur APrioriClassifier, bien qu'il soit très simple (en se basant uniquement sur la classe majoritaire), a des résultats 
# plus faibles en termes de précision et de rappel. En validation, il a une précision de 0.69 et un rappel de 1.0, mais avec beaucoup de faux
# positifs (FP = 62), ce qui montre qu'il est trop optimiste dans la prédiction de la classe positive, sans prendre en compte les relations 
# complexes entre les attributs et la cible. Ce classificateur serait préférable dans des cas où la simplicité est plus importante que la 
# précision des prédictions.

# En résumé, MAP2DClassifier semble être le meilleur choix théorique pour cet exercice, car il parvient à équilibrer à la fois précision et rappel,
# ce qui est crucial dans des scénarios où les deux mesures sont importantes. Cependant, le choix entre ces classificateurs dépendra des priorités 
# spécifiques du problème (par exemple, maximiser la précision ou capturer tous les cas positifs).
#####

# Question 3.1 : complexité en mémoire
def nbParams(df, attrs=None):
    """
    Calcule la taille mémoire nécessaire pour représenter la table de probabilités
    d'un DataFrame donné en supposant que chaque valeur est représentée par un float 
    (8 octets), et que la taille de la table est déterminée par le produit des tailles 
    des attributs.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les variables dont on veut calculer la taille mémoire.
    attrs : list of str, optional
        Liste des colonnes (attributs) du DataFrame pour lesquelles on souhaite
        calculer la taille mémoire. Si None, toutes les colonnes du DataFrame sont utilisées.

    Returns
    -------
    int
        La taille mémoire totale en octets pour représenter la table de probabilités.
    
    Affiche la taille mémoire dans différentes unités (octets, Ko, Mo, Go, To).
    """
    if attrs is None:
        attrs = [col for col in df.columns]
    
    # Taille de chaque attribut (nombre de valeurs uniques)
    attr_sizes = [len(df[attr].unique()) for attr in attrs]
    
    # Calcul de la taille de la table (produit des tailles des attributs)
    table_size = math.prod(attr_sizes)

    # Taille mémoire totale (en octets, avec 8 octets par valeur)
    memory_size = table_size * 8
    
    # Affichage de la taille mémoire en fonction de l'unité
    if memory_size < 1024:
        print(f"{len(attrs)} variable(s) : {memory_size} octets")
    elif memory_size < 1024**2:
        ko = memory_size // 1024
        o = memory_size % 1024
        print(f"{len(attrs)} variable(s) : {memory_size} octets = {ko}ko {o}o")
    elif memory_size < 1024**3:
        mo = memory_size // 1024**2
        ko = (memory_size % 1024**2) // 1024
        print(f"{len(attrs)} variable(s) : {memory_size} octets = {mo}mo {ko}ko {memory_size % 1024}o")
    elif memory_size < 1024**4:
        go = memory_size // 1024**3
        mo = (memory_size % 1024**3) // 1024**2
        ko = (memory_size % 1024**2) // 1024
        print(f"{len(attrs)} variable(s) : {memory_size} octets = {go}go {mo}mo {ko}ko {memory_size % 1024}o")
    else:
        to = memory_size // 1024**4
        print(f"{len(attrs)} variable(s) : {memory_size} octets = {to}to")

    return memory_size

# Question 3.2 : complexité en mémoire sous hypothèse d'indépendance complète
def nbParamsIndep(df, attrs=None):
    """
    Calcule la taille mémoire nécessaire pour représenter les tables de probabilités
    en supposant l'indépendance des variables. Chaque variable est représentée par un
    nombre de valeurs uniques multiplié par 8 octets (taille d'un float).

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les variables dont on veut calculer la taille mémoire.
    attrs : list of str, optional
        Liste des colonnes (attributs) du DataFrame pour lesquelles on souhaite
        calculer la taille mémoire sous l'hypothèse d'indépendance. Si None, toutes les colonnes
        du DataFrame sont utilisées.

    Returns
    -------
    int
        La taille mémoire totale en octets sous l'hypothèse d'indépendance.
    
    Affiche également la taille mémoire totale en octets.
    """
    if attrs is None:
        attrs = [col for col in df.columns]
    
    total_memory = 0
    for attr in attrs:
        # Calcul du nombre de valeurs uniques par attribut
        num_unique_values = len(df[attr].unique())
        
        # Calcul de la mémoire pour cet attribut (8 octets par valeur)
        memory_for_attr = num_unique_values * 8
        total_memory += memory_for_attr
        
    # Affichage de la taille mémoire totale
    print(f"{len(attrs)} variable(s) : {total_memory} octets")
    return total_memory

#####
# QUESTION 3.3.a : Preuve de l'indépendance conditionnelle
#####
# Soit A, B et C trois variables aléatoires. La loi jointe de ces trois variables
# est donnée par : P(A, B, C) = P(A | B, C) * P(B, C)
#
# P(A, B, C) : Il s'agit de la probabilité que les trois événements 
# A, B, et C se produisent simultanément. Cela représente la probabilité conjointe de ces trois variables aléatoires.
#
# P(A | B, C) : C'est la probabilité de l'événement A se produisant sachant que B et C sont déjà survenus, 
# c'est-à-dire la probabilité conditionnelle de A étant donné B et C.
#
# P(B, C) : Cela représente la probabilité que les événements B et C se produisent simultanément. 
# C'est la probabilité conjointe de B et C.
#
# Si A est indépendant de C sachant B, cela signifie que P(A | B, C) = P(A | B).
# En utilisant cette hypothèse, nous pouvons réécrire la loi jointe de la manière suivante :
# P(A, B, C) = P(A | B) * P(B, C)
# Ensuite, on peut factoriser P(B, C) en P(B) * P(C | B), car C conditionné par B
# est indépendant de A. Cela nous donne la factorisation suivante :
# P(A, B, C) = P(A | B) * P(B) * P(C | B)
#
# De plus, on peut utiliser les propriétés des probabilités conditionnelles :
# P(A | B) peut être réécrit comme P(A, B) / P(B), et P(A, B) peut être écrit comme
# P(B | A) * P(A). Ainsi, nous obtenons :
# P(A, B, C) = (P(B | A) * P(A) / P(B)) * P(B) * P(C | B)
# En simplifiant, cela donne :
# P(A, B, C) = P(A) * P(B | A) * P(C | B)
#
# Cette démonstration montre que nous pouvons factoriser la distribution jointe
# sous l'hypothèse d'indépendance conditionnelle.
#####

#####
# QUESTION 3.3.b : Complexité en indépendance conditionnelle
#####
# Supposons que les variables A, B et C ont chacune 5 valeurs possibles. 
# Calculons la taille mémoire nécessaire pour représenter cette distribution
# avec et sans l'utilisation de l'indépendance conditionnelle.
#
# 1. **Sans indépendance conditionnelle**:
# La loi jointe complète P(A, B, C) a une table de taille 5 * 5 * 5 = 125.
# Chaque élément occupe 8 octets. La taille totale est donc :
# Taille mémoire sans indépendance conditionnelle = 125 * 8 = 1000 octets.
#
# 2. **Avec indépendance conditionnelle**:
# On factorise la loi jointe comme suit :
# P(A, B, C) = P(A) * P(B | A) * P(C | B)
# Cela donne trois tables :
# - P(A) avec 5 éléments.
# - P(B | A) avec 5 * 5 = 25 éléments.
# - P(C | B) avec 5 * 5 = 25 éléments.
#
# La taille mémoire totale est donc :
# Taille mémoire avec indépendance conditionnelle = (5 + 25 + 25) * 8 = 55 * 8 = 440 octets.
#
# Conclusion : L'indépendance conditionnelle réduit la taille mémoire de 1000 octets à 440 octets.
#####

#####
# Question 4.1 : Exemples
#####
# 1. A, B, C, D, E complètement indépendantes
# utils.drawGraphHorizontal("A;B;C;D;E")
#
# 2. A, B, C, D, E sans aucune indépendance
# utils.drawGraphHorizontal("A->B->C->D->E")
#####

#####
# QUESTION 4.2 : Naïve Bayes
#####
# Le modèle Naïve Bayes suppose que les attributs sont conditionnellement 
# indépendants les uns des autres étant donné la variable cible (target).
# 
# 1. Décomposition de la vraisemblance :
#    La probabilité conjointe conditionnelle des attributs, sachant target,
#    peut être exprimée comme suit :
#    P(attr1, attr2, attr3, ...∣target) = P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ...
#
# 2. Décomposition de la distribution a posteriori :
#    En utilisant la formule de Bayes, la probabilité a posteriori est donnée par :
#    P(target | attr1, attr2, attr3, ...) =
#        [P(attr1 | target) * P(attr2 | target) * P(attr3 | target) * ... * P(target)] /
#        P(attr1, attr2, attr3, ...)
#    
#    Où :
#      - P(target) est la probabilité a priori de target.
#      - P(attr1, attr2, attr3, ...) est la probabilité conjointe des attributs, calculée comme suit :
#        P(attr1, attr2, attr3, ...) = P(attr1) * P(attr2) * P(attr3) * ... (car 2 attributs sont toujours indépendants conditionnellement à target)
#
# Ce modèle est simpliste mais fonctionne bien dans de nombreux cas pratiques.
#####

# Question 4.3 : modèle graphique et naïve bayes
# Question 4.3.a
def drawNaiveBayes(df, target):
    """
    Dessine un graphique de Naïve Bayes en représentant les relations entre la cible et les attributs.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à visualiser.
    target : str
        Nom de la colonne cible (la variable à prédire) dans le DataFrame.
    
    Returns
    -------
    Graphique représentant le modèle Naïve Bayes avec des arcs entre la cible et les attributs.
    """
    arcs = ";".join([f"{target}->{col}" for col in df.columns if col != target])
    return drawGraph(arcs)

# Question 4.3.b
def nbParamsNaiveBayes(df, target, attrs=None):
    """
    Calcule la taille mémoire nécessaire pour stocker les paramètres d'un classificateur Naïve Bayes.
    
    Cette fonction calcule la mémoire utilisée par les distributions de la cible et les distributions conditionnelles 
    pour chaque attribut. Elle affiche la taille mémoire en octets, kilo-octets, méga-octets ou giga-octets, en fonction du total.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à utiliser pour le calcul de la taille mémoire.
    target : str
        Nom de la colonne cible dans le DataFrame.
    attrs : list of str, optional
        Liste des attributs à considérer. Par défaut, tous les attributs sauf la cible sont pris en compte.
    
    Returns
    -------
    int
        La taille mémoire totale en octets pour stocker les paramètres du modèle Naïve Bayes.
    """
    if attrs is None:
        attrs = [col for col in df.columns if col != target]
        
    if len(attrs) == 0:
        target_memory = len(df[target].unique()) * 8
        print(f"0 variable(s) : {target_memory} octets")
        return target_memory

    attrs = [attr for attr in attrs if attr != target]
    
    # Taille mémoire pour la distribution de target
    target_memory = len(df[target].unique()) * 8

    # Taille mémoire pour les distributions conditionnelles de chaque attribut
    conditional_memory = sum(len(df[target].unique()) * len(df[attr].unique()) * 8 for attr in attrs)

    # Total mémoire
    total_memory = target_memory + conditional_memory

    # Affichage du résultat
    if total_memory < 1024:
        print(f"{len(attrs)+1} variable(s) : {total_memory} octets")
    elif total_memory < 1024**2:
        ko = total_memory // 1024
        o = total_memory % 1024
        print(f"{len(attrs)+1} variable(s) : {total_memory} octets = {ko}ko {o}o")
    elif total_memory < 1024**3:
        mo = total_memory // 1024**2
        ko = (total_memory % 1024**2) // 1024
        print(f"{len(attrs)+1} variable(s) : {total_memory} octets = {mo}mo {ko}ko {total_memory % 1024}o")
    else:
        go = total_memory // 1024**3
        mo = (total_memory % 1024**3) // 1024**2
        print(f"{len(attrs)+1} variable(s) : {total_memory} octets = {go}go {mo}mo")

    return total_memory

# Question 4.4 : Classifieur naïve bayes
class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classificateur Naïve Bayes basé sur une approche apriori.

    Cette classe hérite de `APrioriClassifier` et implémente les méthodes nécessaires pour effectuer des prédictions
    basées sur le modèle Naïve Bayes. Elle utilise des probabilités conditionnelles pour chaque attribut et une probabilité a priori
    pour la classe cible.

    Parameters
    ----------
    train_df: pd.DataFrame
        Le DataFrame d'entraînement contenant les données. Le DataFrame doit contenir une colonne cible appelée "target"
        ainsi que des attributs utilisés pour la prédiction.

    Attributs
    ---------
    attributes: List[str]
        Liste des noms des attributs utilisés pour les prédictions (toutes les colonnes sauf la colonne cible).
    p2d_l: Dict[str, P2D_l]
        Dictionnaire contenant les distributions conditionnelles pour chaque attribut (P(attr | target)).
    prior: Dict[int, float]
        Probabilité a priori pour la cible (classe).
    p_attr: Dict[str, Dict]
        Dictionnaire contenant la distribution de probabilité pour chaque attribut.
    """

    def __init__(self, train_df):
        """
        Initialise le classificateur Naïve Bayes en calculant les distributions conditionnelles pour chaque attribut,
        ainsi que la probabilité a priori pour la cible.

        Parameters
        ----------
        train_df: pd.DataFrame
            Le DataFrame d'entraînement contenant les données.
        """
        super().__init__(train_df)
        self.attributes = [col for col in train_df.columns if col != "target"]
        self.p2d_l = {attr: P2D_l(train_df, attr) for attr in self.attributes}
        self.prior = getPrior(train_df)
        self.p_attr = {attr: train_df[attr].value_counts(normalize=True).to_dict()
                       for attr in self.attributes}

    def estimProbas(self, attrs):
        """
        Estime les probabilités de chaque classe (0 ou 1) en fonction des attributs fournis.

        Parameters
        ----------
        attrs: Dict[str, value]
            Dictionnaire contenant les attributs et leurs valeurs pour lesquels les probabilités doivent être estimées.

        Returns
        -------
        Dict[int, float]
            Dictionnaire contenant les probabilités estimées pour les classes 0 et 1.
        """
        prob_0 = 1
        prob_1 = 1

        for attr, value in attrs.items():
            if attr in self.p2d_l:
                prob_0 *= self.p2d_l[attr][0].get(value, self.p_attr[attr].get(value, 0))
                prob_1 *= self.p2d_l[attr][1].get(value, self.p_attr[attr].get(value, 0))

        return {0: prob_0, 1: prob_1}

    def estimClass(self, attrs):
        """
        Estime la classe (0 ou 1) pour les attributs donnés en fonction des probabilités estimées.

        Parameters
        ----------
        attrs: Dict[str, value]
            Dictionnaire contenant les attributs et leurs valeurs pour lesquels la classe doit être estimée.

        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        probas = self.estimProbas(attrs)
        return 1 if probas[1] > probas[0] else 0

class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classificateur Naïve Bayes basé sur le maximum a posteriori (MAP).

    Cette classe suppose que les attributs sont conditionnellement indépendants les uns des autres étant donné la variable cible.
    Elle implémente un modèle Naïve Bayes avec une estimation des probabilités a posteriori (MAP).

    Parameters
    ----------
    train_df: pd.DataFrame
        Le DataFrame d'entraînement contenant les données. Le DataFrame doit contenir une colonne cible appelée "target"
        ainsi que des attributs utilisés pour la prédiction.

    Attributs
    ---------
    attributes: List[str]
        Liste des noms des attributs utilisés pour les prédictions (toutes les colonnes sauf la colonne cible).
    p2d_l: Dict[str, P2D_l]
        Dictionnaire contenant les distributions conditionnelles pour chaque attribut (P(attr | target)).
    prior: Dict[int, float]
        Probabilité a priori pour la classe cible (target).
    p_attr: Dict[str, Dict]
        Dictionnaire contenant la distribution de probabilité pour chaque attribut.
    """

    def __init__(self, train_df):
        """
        Initialise le classificateur Naïve Bayes basé sur MAP en calculant les distributions conditionnelles pour chaque attribut,
        ainsi que la probabilité a priori pour la cible.

        Parameters
        ----------
        train_df: pd.DataFrame
            Le DataFrame d'entraînement contenant les données.
        """
        super().__init__(train_df)
        self.attributes = [col for col in train_df.columns if col != "target"]

        # Calculer les probabilités conditionnelles P(attr=a | target=t) pour chaque attribut
        self.p2d_l = {attr: P2D_l(train_df, attr) for attr in self.attributes}

        # Calculer les probabilités a priori de target
        self.prior = getPrior(train_df)

        # Calculer les probabilités marginales P(attr) pour chaque attribut
        self.p_attr = {attr: train_df[attr].value_counts(normalize=True).to_dict() for attr in self.attributes}

    def estimProbas(self, attrs):
        """
        Estime les probabilités a posteriori P(target | attr1, attr2, ...) 
        en utilisant le maximum a posteriori (MAP).

        Parameters
        ----------
        attrs: Dict[str, value]
            Dictionnaire représentant un individu avec ses attributs.

        Returns
        -------
        Dict[int, float]
            Dictionnaire contenant les probabilités estimées pour les classes 0 et 1.
        """
        # Initialiser les probabilités avec les priors P(target)
        prob_0 = 1 - self.prior["estimation"]  # P(target=0)
        prob_1 = self.prior["estimation"]  # P(target=1)

        # Calculer la probabilité de chaque attribut P(attr) et les probabilités conditionnelles P(attr|target)
        prob_attr_0 = 1
        prob_attr_1 = 1

        for attr, value in attrs.items():
            if attr in self.p2d_l:
                # Calculer P(attr|target=0) et P(attr|target=1)
                prob_attr_0 *= self.p2d_l[attr].get(0, {}).get(value, 0)
                prob_attr_1 *= self.p2d_l[attr].get(1, {}).get(value, 0)

        # Calculer la normalisation pour P(attr1)*P(attr2)*...
        normalisation = 1
        for attr in attrs:
            if attr in self.p_attr:
                normalisation *= self.p_attr[attr].get(attrs[attr], 1)  # P(attr) pour chaque attribut

        # Calculer les probabilités a posteriori en utilisant la formule MAP
        prob_0 = prob_0 * prob_attr_0 / normalisation
        prob_1 = prob_1 * prob_attr_1 / normalisation

        # Normaliser les probabilités
        total_prob = prob_0 + prob_1
        if total_prob == 0:
            return {0: 0.5, 1: 0.5}  # Cas où les probabilités sont égales (cas pathologique)

        return {0: prob_0 / total_prob, 1: prob_1 / total_prob}

    def estimClass(self, attrs):
        """
        Estime la classe en utilisant les probabilités calculées par estimProbas.

        Parameters
        ----------
        attrs: Dict[str, value]
            Dictionnaire représentant un individu avec ses attributs.

        Returns
        -------
        int
            La classe estimée (0 ou 1).
        """
        probas = self.estimProbas(attrs)
        return 1 if probas[1] > probas[0] else 0

# Question 5.1
def isIndepFromTarget(df, attr, x):
    """
    Vérifie si un attribut est indépendant de la cible dans le dataframe en utilisant le test du chi2.

    Cette fonction effectue un test du chi2 pour vérifier si un attribut donné est indépendant de la variable cible.
    Si la valeur p du test est supérieure ou égale au seuil de signification `x`, l'attribut est considéré comme indépendant
    de la cible.

    Parameters
    ----------
    df: pd.DataFrame
        Le DataFrame contenant les données à tester.
    
    attr: str
        Le nom de l'attribut à tester pour l'indépendance avec la cible.
    
    x: float
        Seuil de signification pour le test chi2. Si la valeur p est supérieure ou égale à `x`, l'attribut est considéré
        comme indépendant de la cible.

    Returns
    -------
    bool
        True si l'attribut est indépendant de la cible au seuil de signification donné, sinon False.
    """
    contingency_table = pd.crosstab(df[attr], df['target'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return p >= x

# Question 5.2
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Classificateur Naïve Bayes réduit utilisant une sélection d'attributs basée sur l'indépendance avec la cible.

    Cette classe hérite de `MLNaiveBayesClassifier` et réduit le nombre d'attributs en fonction de leur indépendance avec la cible
    en utilisant un test du chi2 et un seuil de signification donné.

    Parameters
    ----------
    train_df: pd.DataFrame
        Le DataFrame d'entraînement contenant les données.
    
    significance_level: float
        Niveau de signification utilisé pour tester l'indépendance des attributs vis-à-vis de la cible.

    Attributs
    ---------
    selected_attributes: List[str]
        Liste des attributs sélectionnés après avoir passé le test d'indépendance avec la cible.
    """
    
    def __init__(self, train_df, significance_level):
        """
        Initialise le classificateur en sélectionnant les attributs non indépendants de la cible, puis entraîne
        le classificateur Naïve Bayes sur ces attributs réduits.

        Parameters
        ----------
        train_df: pd.DataFrame
            Le DataFrame d'entraînement contenant les données.
        
        significance_level: float
            Le seuil de signification pour tester l'indépendance des attributs vis-à-vis de la cible.
        """
        self.selected_attributes = [
            attr for attr in train_df.columns if attr != "target" and not isIndepFromTarget(train_df, attr, significance_level)
        ]
        reduced_train_df = train_df[self.selected_attributes + ["target"]]
        super().__init__(reduced_train_df)
    
    def draw(self):
        """
        Dessine un graphique de Naïve Bayes en utilisant les attributs réduits.

        Returns
        -------
        matplotlib.figure.Figure
            Un graphique représentant le modèle Naïve Bayes sur les données réduites.
        """
        reduced_df = self.train_df[self.selected_attributes]
        return drawNaiveBayes(reduced_df, "target")

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    Classificateur MAP Naïve Bayes réduit utilisant une sélection d'attributs basée sur l'indépendance avec la cible.

    Cette classe hérite de `MAPNaiveBayesClassifier` et réduit le nombre d'attributs en fonction de leur indépendance avec la cible
    en utilisant un test du chi2 et un seuil de signification donné.

    Parameters
    ----------
    train_df: pd.DataFrame
        Le DataFrame d'entraînement contenant les données.
    
    significance_level: float
        Niveau de signification utilisé pour tester l'indépendance des attributs vis-à-vis de la cible.

    Attributs
    ---------
    selected_attributes: List[str]
        Liste des attributs sélectionnés après avoir passé le test d'indépendance avec la cible.
    """
    
    def __init__(self, train_df, significance_level):
        """
        Initialise le classificateur en sélectionnant les attributs non indépendants de la cible, puis entraîne
        le classificateur MAP Naïve Bayes sur ces attributs réduits.

        Parameters
        ----------
        train_df: pd.DataFrame
            Le DataFrame d'entraînement contenant les données.
        
        significance_level: float
            Le seuil de signification pour tester l'indépendance des attributs vis-à-vis de la cible.
        """
        self.selected_attributes = [
            attr for attr in train_df.columns if attr != "target" and not isIndepFromTarget(train_df, attr, significance_level)
        ]
        reduced_train_df = train_df[self.selected_attributes + ["target"]]
        super().__init__(reduced_train_df)

    def draw(self):
        """
        Dessine un graphique de Naïve Bayes en utilisant les attributs réduits.

        Returns
        -------
        matplotlib.figure.Figure
            Un graphique représentant le modèle MAP Naïve Bayes sur les données réduites.
        """
        reduced_df = self.train_df[self.selected_attributes]
        return drawNaiveBayes(reduced_df, "target")
        
#####
# Question 6.1
#####
# 1. Le point idéal :
# Le point idéal sur un graphique de précision vs rappel se situe dans le coin supérieur droit, c'est-à-dire :
# 
# Précision = 1.0 (ce qui signifie qu'aucune prédiction fausse positive n'est effectuée),
# Rappel = 1.0 (ce qui signifie que toutes les instances positives sont correctement identifiées).
# Dans ce scénario, le classificateur aurait une précision parfaite et un rappel parfait, c'est-à-dire qu'il fait toutes les prédictions correctement, sans aucun faux positif ni faux négatif.
# 
# 2. Comparer les classificateurs sur un graphique de précision vs rappel :
# Pour comparer les différents classificateurs, nous pouvons utiliser un graphique où :
# 
# L'axe des x représente le précision,
# L'axe des y représente la rappel.
# Chaque classificateur peut être représenté par un point dans cet espace, et chaque point a des coordonnées basées sur ses performances en termes de précision et de rappel. On pourrait procéder par exemple:
# 
# Position du point : Le point de chaque classificateur sur le graphique est déterminé par ses valeurs de précision et de rappel. Plus un point est proche du coin supérieur droit, plus le classificateur est performant.
# Comparer les classificateurs :
# Un classificateur avec un rappel élevé et une précision élevée se rapprochera du point idéal (coin supérieur droit).
# Un classificateur avec un rappel faible et une précision faible se trouvera près du coin inférieur gauche.
# Les compromis : Parfois, il y a un compromis entre précision et rappel. Un classificateur peut avoir une très bonne précision mais un faible rappel, ou vice-versa. 
# En général, si un classificateur est situé plus près du coin supérieur droit, il sera considéré comme meilleur.
#####

def mapClassifiers(dic, df):
    """
    Représente graphiquement les classificateurs dans l'espace précision-rappel avec des limites [0,1] et des 'x' comme marqueurs.

    Cette fonction génère un graphique montrant la précision et le rappel des différents classificateurs dans un
    espace à deux dimensions. Chaque classificateur est représenté par un point dans cet espace, où l'axe des x
    correspond à la précision et l'axe des y au rappel.

    Parameters
    ----------
    dic: Dict[str, object]
        Un dictionnaire où chaque clé est le nom d'un classificateur et chaque valeur est une instance de celui-ci.
    
    df: pd.DataFrame
        Le DataFrame contenant les données à évaluer. Il est utilisé pour calculer les statistiques (précision et rappel)
        pour chaque classificateur.

    Returns
    -------
    None
        La fonction ne retourne rien, elle affiche simplement le graphique de précision-rappel.
    """
    plt.figure(figsize=(8, 6))
    plt.xlabel("Précision")
    plt.ylabel("Rappel")
    plt.title("Précision vs Rappel des Classifieurs")

    for name, classifier in dic.items():
        # Statistiques sur le DataFrame
        stats = classifier.statsOnDF(df)
        precision = stats['Précision']
        rappel = stats['Rappel']

        # Afficher chaque point avec un 'x'
        plt.scatter(precision, rappel, label=name, color='red', marker='x')
        plt.text(precision, rappel, name, fontsize=10)

    # Ajouter une ligne guide au point idéal
    plt.axvline(x=1, color='grey', linestyle='--', linewidth=0.5)
    plt.axhline(y=1, color='grey', linestyle='--', linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.show()

#####
# Question 6.3 : Conclusion
#####
# Cette analyse présente les résultats de sept classificateurs appliqués à un ensemble de données d'entraînement et de test. 
# Les performances des modèles ont été mesurées en termes de précision (Précision) et de rappel (Rappel).

# Résultats pour le classificateur 1 (APrioriClassifier):
# - Test en apprentissage : Précision = 0.745, Rappel = 1.0, avec 404 vrais positifs (VP), 138 faux positifs (FP)
# - Test en validation : Précision = 0.69, Rappel = 1.0, avec 138 VP, 62 FP
# Ce classificateur montre un excellent rappel mais une précision plus faible en validation.

# Résultats pour le classificateur 2 (ML2DClassifier):
# - Test en apprentissage : Précision = 0.846, Rappel = 0.842, avec 340 VP, 76 vrais négatifs (VN), 62 FP, 64 faux négatifs (FN)
# - Test en validation : Précision = 0.807, Rappel = 0.877, avec 121 VP, 33 VN, 29 FP, 17 FN
# Ce classificateur présente un bon compromis entre précision et rappel, tant en apprentissage qu'en validation.

# Résultats pour le classificateur 3 (MAP2DClassifier) :
# - Test en apprentissage : Précision = 0.846, Rappel = 0.842, avec 340 VP, 76 VN, 62 FP, 64 FN
# - Test en validation : Précision = 0.807, Rappel = 0.877, avec 121 VP, 33 VN, 29 FP, 17 FN
# Les résultats sont similaires à ceux du classificateur 2, montrant un bon équilibre entre précision et rappel.

# Résultats pour le classificateur 4 (MAPNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.934, Rappel = 0.946, avec 382 VP, 111 VN, 27 FP, 22 FN
# - Test en validation : Précision = 0.914, Rappel = 0.384, avec 53 VP, 57 VN, 5 FP, 85 FN
# Bien que ce classificateur affiche une bonne précision et un excellent rappel en apprentissage, il montre une forte baisse de performance en validation, avec un rappel très faible.

# Résultats pour le classificateur 5 (MLNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.941, Rappel = 0.866, avec 350 VP, 116 VN, 22 FP, 54 FN
# - Test en validation : Précision = 0.961, Rappel = 0.355, avec 49 VP, 60 VN, 2 FP, 89 FN
# Ce classificateur présente une très bonne précision en validation, mais son rappel est faible, ce qui suggère qu'il privilégie les prédictions négatives.

# Résultats pour le classificateur 6 (ReducedMAPNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.931, Rappel = 0.928, avec 375 VP, 110 VN, 28 FP, 29 FN
# - Test en validation : Précision = 0.898, Rappel = 0.384, avec 53 VP, 56 VN, 6 FP, 85 FN
# Ce classificateur montre des performances similaires à celles du classificateur 5, avec une précision relativement élevée mais un faible rappel en validation.

# Résultats pour le classificateur 7 (ReducedMLNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.943, Rappel = 0.861, avec 348 VP, 117 VN, 21 FP, 56 FN
# - Test en validation : Précision = 0.98, Rappel = 0.355, avec 49 VP, 61 VN, 1 FP, 89 FN
# Comme pour le classificateur 5, ce modèle montre une très bonne précision mais un faible rappel en validation.

# En résumé, les classificateurs 2 et 3 (ML2DClassifier et MAP2DClassifier) semblent offrir un meilleur équilibre entre précision et rappel. En revanche, les classificateurs 4, 5, 6 et 7 ont tendance à privilégier la précision au détriment du rappel, particulièrement en validation.
#####

# Question 7.1 : calcul des informations mutuelles
def MutualInformation(df, target, attr):
    """
    Calcule l'information mutuelle entre une variable cible (target) et un attribut (attr).

    Paramètres
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    target : str
        Le nom de la colonne représentant la variable cible.
    attr : str
        Le nom de la colonne représentant l'attribut.

    Retourne
    -------
    float
        La valeur de l'information mutuelle entre la cible et l'attribut.
    """
    joint_prob = pd.crosstab(df[attr], df[target], normalize=True)
    p_x = joint_prob.sum(axis=1)
    p_y = joint_prob.sum(axis=0)
    mi = 0.0
    for x in p_x.index:
        for y in p_y.index:
            p_xy = joint_prob.at[x, y]
            if p_xy > 0:  # Avoid log(0)
                mi += p_xy * np.log2(p_xy / (p_x[x] * p_y[y]))

    return mi

def ConditionalMutualInformation(df, attr1, attr2, target):
    """
    Calcule l'information mutuelle conditionnelle entre deux attributs, 
    étant donné la variable cible (target).

    Paramètres
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr1 : str
        Le nom de la première colonne représentant un attribut.
    attr2 : str
        Le nom de la deuxième colonne représentant un attribut.
    target : str
        Le nom de la colonne représentant la variable cible.

    Retourne
    -------
    float
        La valeur de l'information mutuelle conditionnelle.
    """
    
    # Calculer les probabilités jointes P(x, y, z)
    joint_prob_xyz = pd.crosstab([df[attr1], df[attr2]], df[target], normalize=True).stack()

    # Calculer les probabilités marginales P(x, z) et P(y, z)
    joint_prob_xz = pd.crosstab(df[attr1], df[target], normalize=True).stack()
    joint_prob_yz = pd.crosstab(df[attr2], df[target], normalize=True).stack()

    # Calculer les probabilités marginales P(x, z) et P(y, z)
    prob_z = df[target].value_counts(normalize=True)

    conditional_mi = 0.0
    for (x, y, z), p_xyz in joint_prob_xyz.items():
        p_xz = joint_prob_xz.get((x, z), 0)
        p_yz = joint_prob_yz.get((y, z), 0)
        p_z = prob_z.get(z, 0)

        # Éviter log(0) en vérifiant que toutes les probabilités sont positives
        if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
            conditional_mi += p_xyz * np.log2((p_z * p_xyz) / (p_xz * p_yz))

    return conditional_mi

# Question 7.2 : calcul de la matrice des poids
def MeanForSymetricWeights(cmis):
    """
    Calcule la moyenne des poids symétriques dans une matrice d'informations mutuelles conditionnelles.

    Paramètres
    ----------
    cmis : numpy.ndarray
        Une matrice carrée contenant les informations mutuelles conditionnelles entre attributs.

    Retourne
    -------
    float
        La moyenne des poids symétriques dans la matrice.
    """

    longueur = len(cmis)
    sum = 0
    for i in range(longueur):
        for j in range(i):
            sum += cmis[i][j]

    return sum / (longueur*(longueur-1)/2)

def SimplifyConditionalMutualInformationMatrix(cmis):
    """
    Simplifie une matrice d'informations mutuelles conditionnelles en supprimant les poids faibles.
    En mettant à zéro les valeurs inférieures à la moyenne des poids symétriques dans la matrice.

    Paramètres
    ----------
    cmis : numpy.ndarray
        Une matrice carrée contenant les informations mutuelles conditionnelles entre attributs.

    Retourne
    -------
    None
        Modifie directement la matrice passée en paramètre.
    """
    moy = MeanForSymetricWeights(cmis)
    longueur = len(cmis)
    for i in range(longueur):
        for j in range(longueur):
            if cmis[i][j] < moy:
                cmis[i][j] = 0

# Question 7.3 : Arbre (forêt) optimal entre les attributs
def find(parent, i):
    """
    Trouve le représentant (racine) d'un élément dans un ensemble disjoint.

    Paramètres
    ----------
    parent : dict
        Un dictionnaire où chaque clé est un élément et la valeur est son parent.
    i : Any
        L'élément pour lequel on souhaite trouver le représentant.

    Retourne
    -------
    Any
        Le représentant (racine) de l'ensemble contenant l'élément i.
    """
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    """
    Fusionne deux ensembles disjoints représentés par leurs racines.

    Paramètres
    ----------
    parent : dict
        Un dictionnaire où chaque clé est un élément et la valeur est son parent.
    rank : dict
        Un dictionnaire où chaque clé est un élément et la valeur est son rang 
        (profondeur approximative de l'arbre).
    x : Any
        La racine du premier ensemble.
    y : Any
        La racine du second ensemble.

    Retourne
    -------
    None
        Modifie directement les dictionnaires `parent` et `rank`.
    """

    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1

def Kruskal(df, cmis):
    """
    Applique l'algorithme de Kruskal pour construire un arbre couvrant minimal à partir 
    d'une matrice d'informations mutuelles conditionnelles.

    Paramètres
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les colonnes représentant les attributs.
    cmis : numpy.ndarray
        Une matrice carrée contenant les informations mutuelles conditionnelles entre attributs.

    Retourne
    -------
    list
        Une liste de tuples représentant les arcs de l'arbre couvrant minimal, 
        sous la forme (attribut1, attribut2, poids).
    """
    longueur = len(cmis)
    arcs = []
    # Créer une liste des arcs avec leur poids
    for i in range(longueur):
        for j in range(i):
            if cmis[i][j] != 0:
                temp_tup = (df.keys()[j], df.keys()[i], cmis[i][j])
                arcs.append(temp_tup)

    # Trier les arcs par poids en ordre décroissant
    arcs = sorted(arcs, key=lambda item: item[2], reverse=True)

    parent = {}
    rank = {}

    # Initialiser les ensembles disjoints pour chaque sommet
    for vertex in df.keys():
        parent[vertex] = vertex
        rank[vertex] = 0

    mst = []

    # Appliquer l'algorithme de Kruskal pour construire l'arbre couvrant minimal
    for arc in arcs:
        u, v, weight = arc
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            mst.append((u, v, np.float64(weight)))
            if len(mst) == 6:
                break
    return mst

# Question 7.4: Orientation des arcs entre attributs.
def ConnexSets(list_arcs):
    """
    Calcule les ensembles connexes d'un graphe représenté par une liste d'arcs.

    Paramètres
    ----------
    list_arcs : list of tuple
        Une liste d'arcs, où chaque arc est représenté par un tuple (noeud1, noeud2).

    Retourne
    -------
    list of set
        Une liste d'ensembles, où chaque ensemble contient les nœuds appartenant à une composante connexe.
    """
    liste_ensembles = []
    for arc in list_arcs:
        added = 0
        for ensemble in liste_ensembles:
            if arc[0] in ensemble or arc[1] in ensemble:
                ensemble.add(arc[0])
                ensemble.add(arc[1])
                added = 1
                break
        if added == 0:
            liste_ensembles.append({arc[0], arc[1]})
    return liste_ensembles

def OrientConnexSets(df, liste_arcs, target):
    """
    Oriente les arcs dans un graphe en fonction de la variable cible et de l'information mutuelle.

    Cette fonction identifie les composantes connexes du graphe, sélectionne une racine 
    pour chaque composante en fonction de l'information mutuelle avec la classe cible, 
    puis oriente les arcs en partant de cette racine en utilisant une recherche en profondeur (DFS).

    Paramètres
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    liste_arcs : list of tuple
        Une liste d'arcs non orientés, où chaque arc est représenté par un tuple (noeud1, noeud2, poids).
    target : str
        Le nom de la colonne représentant la variable cible.

    Retourne
    -------
    list of tuple
        Une liste d'arcs orientés, où chaque arc est représenté par un tuple (parent, enfant).
    """
    # Obtenir les composants connectés (ensembles)
    liste_ensembles = ConnexSets(liste_arcs)
    
    # Calculer l'information mutuelle entre chaque attribut et la classe
    mutual_info = {attr: MutualInformation(df, target, attr) for attr in df.columns if attr != target}
    
    # Initialiser la liste pour stocker les arcs orientés
    oriented_arcs = []
    
    # Itérer sur chaque ensemble et orienter les arcs
    for ensemble in liste_ensembles:
        # Sélectionner la racine en fonction de l'information mutuelle la plus élevée
        root = max(ensemble, key=lambda x: mutual_info[x])
        
        # Créer un graphe sous forme de liste d'adjacence (graphe non orienté)
        graph = {node: [] for node in ensemble}
        for arc in liste_arcs:
            node1, node2, _ = arc
            
            if node1 in ensemble and node2 in ensemble:
                graph[node1].append(node2)
                graph[node2].append(node1)
        
        # Effectuer une recherche en profondeur (DFS) pour orienter les arcs
        visited = set()
        
        def dfs(node, parent):
            # Visiter le nœud courant et orienter l'arc du parent vers le nœud
            if node in visited:
                return
            visited.add(node)
            
            if parent is not None:
                oriented_arcs.append((parent, node))  # Orienter l'arc du parent vers le nœud courant
            
            # Explorer tous les voisins du nœud courant
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, node)  # Visiter récursivement les voisins

        # Démarrer la recherche en profondeur depuis la racine
        dfs(root, None)
    return oriented_arcs

# Question 7.5: Classifieur TAN
def P_cond(attr, value, parent, parent_value, df):
    """
    Calculer la probabilité conditionnelle P(attr=value | parent=parent_value).

    Paramètres
    ----------
    attr : str
        Le nom de l'attribut (enfant).
    value : Any
        La valeur de l'attribut (enfant).
    parent : str
        Le nom de l'attribut parent.
    parent_value : Any
        La valeur de l'attribut parent.
    df : pandas.DataFrame
        Le DataFrame contenant les données.

    Retourne
    -------
    float
        La probabilité conditionnelle. Retourne une petite valeur (1e-10) si aucune donnée pour la paire parent-enfant.
    """
    subset = df[df[parent] == parent_value]  # Filtrer les données pour la valeur spécifique du parent
    count = subset[subset[attr] == value].shape[0]  # Compter les occurrences de la valeur de l'enfant
    total = subset.shape[0]  # Total des occurrences de la valeur du parent
    return count / total if total > 0 else 0  # Retourner la probabilité conditionnelle ou une petite valeur si aucune occurrence

class MAPTANClassifier(APrioriClassifier):
    """
    Classificateur Tree-Augmented Naïve Bayes (TAN) basé sur l'information mutuelle conditionnelle.
    """
    
    def __init__(self, train_df):
        super().__init__(train_df)
        self.attributes = [col for col in train_df.columns if col != "target"]
        
        # Calculer les probabilités conditionnelles P(attr=a | target=t) pour chaque attribut
        self.p2d_l = {attr: P2D_l(train_df, attr) for attr in self.attributes}
        
        # Calculer les probabilités a priori de target
        self.prior = getPrior(train_df)
        
        # Calculer les probabilités marginales P(attr) pour chaque attribut
        self.p_attr = {attr: train_df[attr].value_counts(normalize=True).to_dict() for attr in self.attributes}
        
        # Initialize arcs and other TAN-specific structures
        self.cmis = np.array([[0 if x == y else ConditionalMutualInformation(self.train_df, x, y, "target") 
                               for x in self.train_df.keys() if x != "target"] for y in self.train_df.keys() if y != "target"])
        SimplifyConditionalMutualInformationMatrix(self.cmis)
        self.liste_arcs = Kruskal(self.train_df, self.cmis)
        self.oriented_liste_arcs = OrientConnexSets(self.train_df, self.liste_arcs, 'target')
        self.arcs = ";".join([f"{'target'}->{col}" for col in self.train_df.columns if col != 'target'])
        self.arcs += ";" + ";".join([f"{u}->{v}" for u, v in self.oriented_liste_arcs])

    def estimProbas(self, attrs):
        """
        Estimate posterior probabilities P(target=0|attrs) and P(target=1|attrs)
        using TAN structure and MAP.
        """
        # Initial probabilities based on prior
        prob_0 = 1 - self.prior["estimation"]  # P(target=0)
        prob_1 = self.prior["estimation"]      # P(target=1)
    
        prob_attr_0 = 1
        prob_attr_1 = 1
    
        # Create a mapping of parent-child relationships
        parent_map = {child: parent for parent, child in self.oriented_liste_arcs}
    
        # Loop through each attribute in the dataset
        for attr in self.attributes:
            attr_value = attrs.get(attr, None)
            if attr_value is None:
                continue  # Skip if the attribute is missing from the input
    
            if attr not in parent_map:
                # Root attribute: P(attr | target)
                prob_attr_0 *= self.p2d_l[attr][0].get(attr_value, 0)  # P(attr=value | target=0)
                prob_attr_1 *= self.p2d_l[attr][1].get(attr_value, 0)  # P(attr=value | target=1)
            else:
                # Dependent attribute: P(attr | parent, target)
                parent = parent_map[attr]
                parent_value = attrs.get(parent, None)
                if parent_value is not None:
                    # Calculate conditional probability P(attr | parent, target)
                    prob_attr_0 *= P_cond(attr, attr_value, parent, parent_value, self.train_df) * self.p2d_l[parent][0].get(parent_value, 0)
                    prob_attr_1 *= P_cond(attr, attr_value, parent, parent_value, self.train_df) * self.p2d_l[parent][1].get(parent_value, 0)
    
        # Apply normalization using the marginal probabilities P(attr)
        normalisation = 1
        for attr in attrs:
            if attr in self.p_attr:
                # P(attr) for each attribute (marginal)
                normalisation *= self.p_attr[attr].get(attrs[attr], 1)  # Default to 1 if attribute value is missing
        
        # Apply the normalization factor, but avoid suppressing the probabilities
        if normalisation > 0:
            prob_0 *= prob_attr_0 / normalisation
            prob_1 *= prob_attr_1 / normalisation
        
        # Normalize probabilities to ensure they sum to 1
        total_prob = prob_0 + prob_1
        if total_prob == 0:
            return {0: 0.5, 1: 0.5}  # If probabilities are zero, return equal probabilities to avoid division by zero
    
        # Return normalized probabilities
        return {0: prob_0 / total_prob, 1: prob_1 / total_prob}

    def estimClass(self, attrs):
        """
        Estime la classe en utilisant les probabilités calculées par estimProbas.

        Args:
            attrs (dict): Un dictionnaire représentant un individu avec ses attributs.

        Returns:
            int: La classe estimée (0 ou 1).
        """
        probas = self.estimProbas(attrs)
        return 1 if probas[1] > probas[0] else 0
        
    def draw(self):
        """
        Visualiser la structure de l'arbre du modèle TAN en tant que graphe.
        """
        return drawGraph(self.arcs)  # Visualize the TAN structure

##### 
# Question 8- Conclusion finale
##### 
# D'après l'analyse de Q7.5, elle présente les résultats de huit classificateurs appliqués à un ensemble de données d'entraînement et de test. 
# Les performances des modèles ont été mesurées en termes de précision (Précision) et de rappel (Rappel).

# Résultats pour le classificateur 1 (APrioriClassifier):
# - Test en apprentissage : Précision = 0.745, Rappel = 1.0, avec 404 vrais positifs (VP), 138 faux positifs (FP)
# - Test en validation : Précision = 0.69, Rappel = 1.0, avec 138 VP, 62 FP
# Ce classificateur montre un excellent rappel mais une précision plus faible en validation.

# Résultats pour le classificateur 2 (ML2DClassifier):
# - Test en apprentissage : Précision = 0.846, Rappel = 0.842, avec 340 VP, 76 vrais négatifs (VN), 62 FP, 64 faux négatifs (FN)
# - Test en validation : Précision = 0.807, Rappel = 0.877, avec 121 VP, 33 VN, 29 FP, 17 FN
# Ce classificateur présente un bon compromis entre précision et rappel, tant en apprentissage qu'en validation.

# Résultats pour le classificateur 3 (MAP2DClassifier) :
# - Test en apprentissage : Précision = 0.846, Rappel = 0.842, avec 340 VP, 76 VN, 62 FP, 64 FN
# - Test en validation : Précision = 0.807, Rappel = 0.877, avec 121 VP, 33 VN, 29 FP, 17 FN
# Les résultats sont similaires à ceux du classificateur 2, montrant un bon équilibre entre précision et rappel.

# Résultats pour le classificateur 4 (MAPNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.934, Rappel = 0.946, avec 382 VP, 111 VN, 27 FP, 22 FN
# - Test en validation : Précision = 0.914, Rappel = 0.384, avec 53 VP, 57 VN, 5 FP, 85 FN
# Bien que ce classificateur affiche une bonne précision et un excellent rappel en apprentissage, il montre une forte baisse de performance en validation, avec un rappel très faible.

# Résultats pour le classificateur 5 (MLNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.941, Rappel = 0.866, avec 350 VP, 116 VN, 22 FP, 54 FN
# - Test en validation : Précision = 0.961, Rappel = 0.355, avec 49 VP, 60 VN, 2 FP, 89 FN
# Ce classificateur présente une très bonne précision en validation, mais son rappel est faible, ce qui suggère qu'il privilégie les prédictions négatives.

# Résultats pour le classificateur 6 (ReducedMAPNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.931, Rappel = 0.928, avec 375 VP, 110 VN, 28 FP, 29 FN
# - Test en validation : Précision = 0.898, Rappel = 0.384, avec 53 VP, 56 VN, 6 FP, 85 FN
# Ce classificateur montre des performances similaires à celles du classificateur 5, avec une précision relativement élevée mais un faible rappel en validation.

# Résultats pour le classificateur 7 (ReducedMLNaiveBayesClassifier) :
# - Test en apprentissage : Précision = 0.943, Rappel = 0.861, avec 348 VP, 117 VN, 21 FP, 56 FN
# - Test en validation : Précision = 0.98, Rappel = 0.355, avec 49 VP, 61 VN, 1 FP, 89 FN
# Comme pour le classificateur 5, ce modèle montre une très bonne précision mais un faible rappel en validation.

# Résultats pour le classificateur 8 (MAPTANClassifier) :
# - Test en apprentissage : Précision = 0.939, Rappel = 0.998, avec 403 VP, 112 VN, 26 FP, 1 FN
# - Test en validation : Précision = 0.927, Rappel = 0.920, avec 127 VP, 52 VN, 10 FP, 11 FN
# Ce classificateur se distingue par une excellente performance, tant en apprentissage qu'en validation, avec un très bon équilibre entre précision et rappel. Il affiche de bons résultats de probabilité et de classification pour les individus, avec des prédictions très précises sur les classes.

# En résumé, le classificateur MAPTANClassifier offre les meilleures performances globales pour plusieurs raisons :
# - En apprentissage, il maintient un excellent équilibre entre précision (Précision = 0.939) et rappel (Rappel = 0.998), ce qui indique qu'il parvient à bien classer les instances positives tout en évitant un grand nombre de faux négatifs.
# - En validation, bien que la précision soit légèrement inférieure (Précision = 0.927), le rappel reste très élevé (Rappel = 0.920), ce qui montre qu'il généralise bien aux nouvelles données.
# - Il surpasse les autres classificateurs qui privilégient souvent la précision au détriment du rappel, ce qui entraîne une performance médiocre en validation, comme on le voit pour les classificateurs 4, 5, 6 et 7.
# En outre, la structure de l'algorithme MAPTAN, qui combine l'estimation des probabilités conditionnelles et les arcs orientés du réseau, semble offrir une meilleure capacité à gérer les relations complexes entre les attributs, améliorant ainsi la qualité des prédictions.

# En conclusion, le classificateur MAPTANClassifier est le plus performant dans ce contexte, offrant une robustesse et une précision exceptionnelles sur les données d'apprentissage et de validation.
#####