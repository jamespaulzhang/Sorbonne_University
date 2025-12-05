# Yuxiang ZHANG 21202829
# Kenan Alsafadi 21502362

import numpy as np
import matplotlib.pyplot as plt

def learnHMM(allX, allS, N, K):
    """
    Apprend les paramètres d'un HMM à partir de séquences d'observations et d'états
    
    Paramètres:
    allX -- Séquence(s) d'observations (ou liste de séquences)
    allS -- Séquence(s) d'états cachés (ou liste de séquences) 
    N -- Nombre d'états cachés
    K -- Nombre d'observations possibles
    
    Retourne:
    A -- Matrice de probabilités de transition (N x N)
    B -- Matrice de probabilités d'émission (N x K)
    """
    
    # Initialisation des matrices de comptage avec des zéros
    # selon les spécifications du TME
    A_count = np.zeros((N, N))  # Matrice de comptage des transitions
    B_count = np.zeros((N, K))  # Matrice de comptage des émissions
    
    # Vérification et conversion des entrées en listes si nécessaire
    # Permet de traiter à la fois des séquences simples et des listes de séquences
    if not isinstance(allX, list):
        allX = [allX]
    if not isinstance(allS, list):
        allS = [allS]
    
    # Parcours de toutes les paires (séquence d'observations, séquence d'états)
    for X, S in zip(allX, allS):
        T = len(X)  # Longueur de la séquence courante
        
        # Comptage des transitions entre états consécutifs
        for t in range(T - 1):
            current_state = S[t]      # État au temps t
            next_state = S[t + 1]     # État au temps t+1
            A_count[current_state, next_state] += 1  # Incrémentation du compteur
        
        # Comptage des émissions (observations pour chaque état)
        for t in range(T):
            current_state = S[t]      # État au temps t
            current_obs = X[t]        # Observation au temps t
            B_count[current_state, current_obs] += 1  # Incrémentation du compteur
    
    # Normalisation des matrices de comptage pour obtenir des probabilités
    # Chaque ligne est normalisée pour sommer à 1 (distribution de probabilité)
    A = A_count / A_count.sum(axis=1, keepdims=True)  # Matrice de transition normalisée
    B = B_count / B_count.sum(axis=1, keepdims=True)  # Matrice d'émission normalisée
    
    # Application des contraintes structurelles du modèle HMM pour l'annotation génique
    # Ces contraintes reflètent les règles biologiques des transitions entre états
    A[0, 2] = 0  # Transition impossible : intergénique → position 2 codon
    A[0, 3] = 0  # Transition impossible : intergénique → position 3 codon  
    A[1, 0] = 0  # Transition impossible : position 0 codon → intergénique
    A[1, 1] = 0  # Transition impossible : position 0 codon → position 0 codon
    A[1, 3] = 0  # Transition impossible : position 0 codon → position 3 codon
    A[2, 0] = 0  # Transition impossible : position 1 codon → intergénique
    A[2, 1] = 0  # Transition impossible : position 1 codon → position 0 codon
    A[2, 2] = 0  # Transition impossible : position 1 codon → position 1 codon
    A[3, 2] = 0  # Transition impossible : position 2 codon → position 2 codon
    A[3, 3] = 0  # Transition impossible : position 2 codon → position 3 codon
    
    # Renormalisation après application des contraintes
    # Assure que chaque ligne de A somme à 1 malgré les probabilités forcées à 0
    A = A / A.sum(axis=1, keepdims=True)
    
    return A, B

'''
Question 1 : pour effectuer le comptage, discuter la pertinence d'une initialisation avec des 1 au lieu de 0 pour avoir une estimation robuste (cf TME 6)

Réponse :
Dans le TME6, nous avons déjà discuté que si nous initialisons la matrice de comptage avec des 0 et qu'un état n'apparaît jamais dans les données d'entraînement, nous rencontrons des problèmes lors de la normalisation : soit une division par zéro se produit, soit nous obtenons une distribution de probabilité invalide (la somme de la ligne n'est pas égale à 1).
Cependant, dans le problème d'annotation de gènes du TME7, l'initialisation avec des 0 est appropriée pour les raisons suivantes :

1. Suffisance des données : Dans les données de séquences génétiques, les 4 états (région intergénique, position de codon 0, 1, 2) apparaissent tous dans les données d'entraînement, donc aucun état complètement non observé ne se présente.
2. Contraintes structurelles du modèle : Selon la structure du modèle de Markov caché, certaines transitions sont strictement impossibles (par exemple, passer directement de la région intergénique à la position 2 du codon). L'initialisation avec des 0 permet de refléter précisément ces contraintes structurelles.
3. Estimation précise : Pour des données aussi suffisantes, l'initialisation avec des 0 fournit une estimation plus précise des probabilités, sans introduire de biais via le lissage.
4. Cohérence avec les résultats attendus : L'énoncé demande explicitement une initialisation avec des 0 et fournit les résultats attendus basés sur cette initialisation.
5. Justification biologique : Dans la structure des gènes, certaines transitions sont effectivement impossibles, et il est raisonnable de les représenter par une probabilité nulle.

En revanche, l'initialisation avec des 1 (lissage de Laplace) est plus adaptée dans les cas suivants :
- Données rares
- Possibilité d'états ou de transitions non observés
- Nécessité d'une meilleure généralisation du modèle sur des données non vues

Mais dans ce problème, en raison de la suffisance des données et de la structure claire du modèle, l'initialisation avec des 0 est le choix le plus approprié, permettant d'obtenir une estimation des paramètres plus conforme à la réalité biologique.
Ainsi, nous choisissons d'initialiser les matrices A et B avec des 0 comme demandé dans l'énoncé.
'''

def viterbi(allx, Pi, A, B, epsilon=1e-12):
    """
    Implémentation de l'algorithme de Viterbi pour trouver la séquence d'états
    la plus probable étant donné une séquence d'observations et un modèle HMM
    
    Paramètres:
    ----------
    allx : array (T,)
        Séquence d'observations de longueur T
    Pi: array (N,)
        Distribution de probabilité initiale des états
    A : array (N, N)
        Matrice de transition entre états
    B : array (N, M)
        Matrice d'émission (probabilités d'observation pour chaque état)
    epsilon : float, optionnel
        Petite valeur pour éviter les probabilités nulles (lissage)

    Retourne:
    -------
    best_path : array (T,)
        Séquence d'états la plus probable
    """
    
    # Nombre d'états et longueur de la séquence
    N = len(A)
    T = len(allx)
    
    # Initialisation des matrices delta et psi
    # delta stocke les probabilités maximales accumulées
    # psi stocke les états précédents qui maximisent la probabilité
    delta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)
    
    # Application d'un lissage pour éviter les probabilités nulles
    Pi_smooth = Pi + epsilon
    Pi_smooth = Pi_smooth / np.sum(Pi_smooth)
    
    A_smooth = A + epsilon
    A_smooth = A_smooth / np.sum(A_smooth, axis=1, keepdims=True)
    
    B_smooth = B + epsilon
    B_smooth = B_smooth / np.sum(B_smooth, axis=1, keepdims=True)
    
    # ÉTAPE 1: INITIALISATION
    # Calcul des probabilités initiales pour le premier pas de temps
    # On utilise les logarithmes pour éviter les problèmes numériques
    for i in range(N):
        delta[i, 0] = np.log(Pi_smooth[i]) + np.log(B_smooth[i, allx[0]])
        psi[i, 0] = -1  # Valeur spéciale indiquant l'état initial
    
    # Affichage des valeurs initiales de delta
    print(delta[:, 0])
    
    # ÉTAPE 2: RÉCURSION
    # Pour chaque pas de temps suivant
    for t in range(1, T):
        # Pour chaque état possible au temps t
        for j in range(N):
            # Recherche du maximum parmi tous les états précédents
            max_val = -np.inf
            best_prev_state = -1
            
            for i in range(N):
                # Calcul du score: probabilité accumulée + transition
                score = delta[i, t-1] + np.log(A_smooth[i, j])
                if score > max_val:
                    max_val = score
                    best_prev_state = i
            
            # Mise à jour de delta et psi
            delta[j, t] = max_val + np.log(B_smooth[j, allx[t]])
            psi[j, t] = best_prev_state
        
        # Affichage des valeurs de delta à des intervalles spécifiques
        if t == 100000 or t == 200000 or t == 300000 or t == 400000:
            print(f"t= {t} delta[:,t]= {delta[:, t]}")
    
    # ÉTAPE 3: TERMINAISON
    # Recherche de l'état final le plus probable
    best_log_prob = -np.inf
    best_last_state = -1
    
    for i in range(N):
        if delta[i, T-1] > best_log_prob:
            best_log_prob = delta[i, T-1]
            best_last_state = i
    
    # ÉTAPE 4: RECONSTRUCTION DU CHEMIN (BACKTRACKING)
    # Reconstruction de la séquence d'états optimale en remontant les backpointers
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_last_state
    
    for t in range(T-2, -1, -1):
        best_path[t] = psi[best_path[t+1], t+1]
    
    # Retourne uniquement le meilleur chemin
    return best_path

def get_and_show_coding(etat_predits, annotation_test):
    """
    Convertit les séquences d'états en séquences binaires codant/non-codant
    et prépare la visualisation comparative
    
    Paramètres:
    ----------
    etat_predits : array
        Séquence d'états prédits par le modèle (valeurs: 0,1,2,3)
    annotation_test : array
        Séquence d'états réels de référence (valeurs: 0,1,2,3)
    
    Retourne:
    -------
    codants_predits : array
        Séquence binaire des régions codantes prédites (0=non-codant, 1=codant)
    codants_test : array
        Séquence binaire des régions codantes réelles (0=non-codant, 1=codant)
    """
    
    # Conversion des états en indicateurs binaires codant/non-codant
    # État 0 = non-codant (reste 0)
    # États 1,2,3 = codants (deviennent 1)
    codants_predits = np.where(etat_predits == 0, 0, 1)
    codants_test = np.where(annotation_test == 0, 0, 1)
    
    # Visualisation comparative des séquences codantes/non-codantes
    fig, ax = plt.subplots(figsize=(15, 2))
    ax.plot(codants_predits[100000:200000], label="prediction", ls="--")
    ax.plot(codants_test[100000:200000], label="annotation", lw=3, color="black", alpha=.4)
    plt.legend(loc="best")
    plt.show()
    
    return codants_predits, codants_test

def create_confusion_matrix(y_true, y_pred):
    """
    Calcule la matrice de confusion pour un problème de classification binaire
    
    Paramètres:
    ----------
    y_true : array
        Séquence des états réels (0=non-codant, 1=codant)
    y_pred : array
        Séquence des états prédits (0=non-codant, 1=codant)
    
    Retourne:
    -------
    mat_conf : array (2x2)
        Matrice de confusion sous la forme:
        [[TN, FP],
         [FN, TP]]
    """
    
    # Initialisation de la matrice de confusion 2x2
    mat_conf = np.zeros((2, 2))
    
    # Calcul des composantes de la matrice de confusion
    # TP (True Positive): réel=1, prédit=1
    mat_conf[0, 0] = np.sum((y_true == 1) & (y_pred == 1))

    # FP (False Positive): réel=0, prédit=1
    mat_conf[0, 1] = np.sum((y_true == 0) & (y_pred == 1))

    # TN (True Negative): réel=0, prédit=0
    mat_conf[1, 1] = np.sum((y_true == 0) & (y_pred == 0))

    # FN (False Negative): réel=1, prédit=0
    mat_conf[1, 0] = np.sum((y_true == 1) & (y_pred == 0))
    
    return mat_conf

'''
Question 2 : Donner une interprétation.

Réponse :
À partir de la matrice de confusion obtenue :
mat_conf = array([[202819., 152699.],
[ 31460., 113022.]])

Nous pouvons analyser en détail les performances du modèle :
Métriques de performance clés :
- Vrais Positifs (TP) = 202 819 - régions codantes correctement identifiées
- Faux Positifs (FP) = 152 699 - régions non-codantes incorrectement prédites comme codantes
- Faux Négatifs (FN) = 31 460 - régions codantes manquées (prédites comme non-codantes)
- Vrais Négatifs (TN) = 113 022 - régions non-codantes correctement identifiées
- Exactitude globale = 63,17%

Pour une évaluation plus nuancée, nous calculons également :
- Précision = TP / (TP + FP) = 202819 / (202819 + 152699) ≈ 0,57 (57%)
- Rappel = TP / (TP + FN) = 202819 / (202819 + 31460) ≈ 0,87 (87%)

Interprétation de ces métriques :
- La précision relativement faible (57%) indique que parmi toutes les régions prédites comme codantes, seulement 57% sont véritablement codantes. Le modèle génère donc beaucoup de fausses alarmes.
- Le rappel élevé (87%) révèle que le modèle capture efficacement la majorité des véritables régions codantes, montrant une bonne sensibilité de détection.

Ce profil de performance (rappel élevé mais précision modérée) suggère que le modèle est très sensible pour identifier les régions codantes, mais au prix d'un nombre important de prédictions erronées.

Question 3 : Le modèle peut-il être utilisé pour prédire la position des gènes dans le génôme ?

Réponse nuancée : Le modèle présente une utilité limitée mais réelle :

Points forts :
- Capacité à détecter 87% des vraies régions codantes
- Exactitude supérieure à une classification aléatoire (50%)
- Potentiel comme outil de criblage préliminaire

Limitations critiques :
- Taux de faux positifs très élevé (43% des prédictions codantes sont erronées)
- Exactitude globale insuffisante pour une annotation précise
- Nécessite une validation expérimentale systématique

Conclusion : Ce modèle démontre l'intérêt des chaînes de Markov cachées pour l'analyse de séquences biologiques, mais ses performances actuelles le limitent à un rôle d'outil complémentaire plutôt qu'à une solution autonome pour l'annotation génique précise. Les taux élevés de faux positifs rendent nécessaire une validation supplémentaire pour toute application concrète.
'''

def create_seq(N, Pi, A, B, states, obs):
    """
    Génère une séquence d'états cachés et une séquence d'observations
    à partir des paramètres d'un modèle de Markov caché
    
    Paramètres:
    ----------
    N : int
        Longueur de la séquence à générer
    Pi : array
        Distribution de probabilité initiale des états
    A : array (N_états, N_états)
        Matrice de transition entre états
    B : array (N_états, N_observations)
        Matrice d'émission (probabilités d'observation pour chaque état)
    states : list
        Liste des états possibles
    obs : list
        Liste des observations possibles
    
    Retourne:
    -------
    state_seq : list
        Séquence d'états générée
    obs_seq : list
        Séquence d'observations générée
    """
    
    # Initialisation des séquences
    state_seq = []
    obs_seq = []
    
    # Sélection de l'état initial selon la distribution Pi
    current_state = np.random.choice(states, p=Pi)
    
    # Génération de la séquence de longueur N
    for i in range(N):
        # Ajout de l'état courant à la séquence d'états
        state_seq.append(current_state)
        
        # Génération d'une observation selon la distribution d'émission de l'état courant
        # La distribution d'émission pour l'état courant est la ligne current_state de B
        current_obs = np.random.choice(obs, p=B[current_state])
        obs_seq.append(current_obs)
        
        # Affichage de l'état et de l'observation courants (format demandé)
        print(f"{current_state} {current_obs}")
        
        # Transition vers le prochain état selon la matrice de transition A
        # La distribution de transition pour l'état courant est la ligne current_state de A
        current_state = np.random.choice(states, p=A[current_state])
    
    return state_seq, obs_seq

def get_annoatation2(annotation):
    """
    Transforme les annotations à 4 états en annotations à 10 états
    pour prendre en compte explicitement les codons start et stop
    
    Paramètres:
    ----------
    annotation : array
        Séquence d'annotations originales avec 4 états
        (0=intergénique, 1=position0, 2=position1, 3=position2)
    
    Retourne:
    -------
    annotation2 : array
        Séquence d'annotations transformées avec 10 états:
        0=intergénique, 1-3=codon start, 4-6=codons internes, 7-9=codon stop
    """
    
    # Initialisation de la nouvelle annotation avec des zéros
    annotation2 = np.zeros_like(annotation)
    
    # Parcours de la séquence d'annotation originale
    i = 0
    while i < len(annotation):
        # Si on est dans une région intergénique, on conserve l'état 0
        if annotation[i] == 0:
            annotation2[i] = 0
            i += 1
            continue
        
        # Détection du début d'un gène (codon start)
        # Un gène commence toujours par la séquence d'états 1,2,3
        if i + 2 < len(annotation) and annotation[i] == 1 and annotation[i+1] == 2 and annotation[i+2] == 3:
            # Attribution des états du codon start (1,2,3)
            annotation2[i] = 1    # Première position du codon start
            annotation2[i+1] = 2  # Deuxième position du codon start
            annotation2[i+2] = 3  # Troisième position du codon start
            i += 3
            
            # Traitement des codons internes jusqu'au codon stop
            while i < len(annotation) and annotation[i] != 0:
                # Vérification si on a atteint la fin du gène (codon stop)
                # Un codon stop est suivi d'une région intergénique
                if i + 2 >= len(annotation) or annotation[i+2] == 0 or (
                    i + 3 < len(annotation) and annotation[i+3] == 0):
                    # Attribution des états du codon stop (7,8,9)
                    if i + 2 < len(annotation):
                        annotation2[i] = 7    # Première position du codon stop
                        annotation2[i+1] = 8  # Deuxième position du codon stop
                        annotation2[i+2] = 9  # Troisième position du codon stop
                    i += 3
                    break
                else:
                    # Attribution des états des codons internes (4,5,6)
                    annotation2[i] = 4    # Première position du codon interne
                    annotation2[i+1] = 5  # Deuxième position du codon interne
                    annotation2[i+2] = 6  # Troisième position du codon interne
                    i += 3
        else:
            # Si le motif n'est pas reconnu, on conserve l'état original
            annotation2[i] = annotation[i]
            i += 1
    
    return annotation2

'''
Question 4 : Pour le codon start, on sait que les proportions sont les suivantes: ATG : 83% GTG: 14%, et TTG: 3%. Vérfier dans votre modèle appris que cela correspond approximativement sur la base d'apprentissage.

Réponse :
En analysant la matrice d'émission B de notre modèle appris, nous pouvons examiner les probabilités d'émission pour les états du codon start (états 1, 2, 3) :

État 1 (première position du codon start) :
- A: 83.26% ≈ 83% ✓
- T: 0.84%
- C: 12.13% 
- G: 3.77%

État 2 (deuxième position du codon start) :
- A: 0%
- T: 0.84%
- C: 2.09%
- G: 97.07%

État 3 (troisième position du codon start) :
- A: 1.26%
- T: 0.42%
- C: 97.91%
- G: 0.42%

Analyse :
- La première position montre une forte probabilité pour A (83.26%), ce qui correspond bien à la proportion attendue de 83% pour ATG
- Cependant, les autres positions ne reflètent pas parfaitement les motifs GTG (14%) et TTG (3%)
- Ce résultat suggère que le modèle a partiellement appris le motif du codon start, mais avec certaines limitations

Question 5 : Évaluez les performances du nouveau modèle en faisant de nouvelles prections de genome pour genome_test, et comparez les avec le modèle précédent. 

Réponse :
Comparaison des performances :
| Métrique | Modèle 4 états | Modèle 10 états | Amélioration |
|----------|----------------|-----------------|--------------|
| Exactitude | 63.17% | 63.41% | +0.24% |
| Vrais Positifs (TP) | 202,819 | 211,133 | +8,314 |
| Vrais Négatifs (TN) | 113,022 | 105,895 | -7,127 |

Interprétation :
1. Amélioration modeste : Le nouveau modèle montre une légère amélioration de l'exactitude (+0.24%)
2. Meilleure détection des codants : L'augmentation des vrais positifs (TP) indique que le modèle identifie mieux les régions codantes
3. Légère diminution des vrais négatifs : La baisse des TN suggère que le modèle a un peu plus de difficulté à identifier correctement les régions non-codantes

Conclusion :
Le modèle à 10 états, bien que plus complexe biologiquement, n'apporte qu'une amélioration limitée des performances. Cette amélioration modeste peut s'expliquer par :
- La complexité accrue du modèle nécessitant plus de données d'entraînement
- La difficulté d'estimer précisément les paramètres pour 10 états
- Le fait que les informations supplémentaires sur les codons start/stop ne capturent pas toute la complexité des séquences génomiques

Le modèle reste utile comme outil de prédiction préliminaire, mais des approches supplémentaires seraient nécessaires pour une annotation génique précise.
'''