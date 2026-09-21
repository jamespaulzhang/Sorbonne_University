# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import matplotlib.pyplot as plt
import time
import random
import statistics

"""Mise en œuvre"""
# 3.1 Implémentation

# Entrées-Sorties
def lire_entree(fichier):
    """
    Lit les entrées depuis un fichier texte.

    :param fichier: Chemin du fichier contenant les données.
    :return: Tuple (S, k, V) où :
        - S est la quantité totale.
        - k est le nombre de types de bocaux.
        - V est une liste des capacités triées.
    """
    with open(fichier, 'r') as f:
        lignes = f.readlines()

    # Lire la quantité totale S
    S = int(lignes[0].strip())

    # Lire le nombre de types de bocaux k
    k = int(lignes[1].strip())

    # Lire les capacités V
    V = sorted([int(ligne.strip()) for ligne in lignes[2:]])

    return S, k, V

# Fonctions réalisées avec 3 algorithmes d'après la partie théorique
# Algorithme I
def m(s, i, V):
    """
    Fonction récursive pour calculer le nombre minimal de bocaux nécessaires.

    :param s: La capacité totale à atteindre.
    :param i: L'index du dernier type de bocal à considérer (1-indexé).
    :param V: La liste des capacités des bocaux (1-indexé, avec V[0] ignoré).
    :return: Le nombre minimal de bocaux nécessaires ou +∞ si impossible.
    """
    # Cas de base : capacité négative
    if s < 0:
        return float('inf')  # +∞, impossible d'atteindre une capacité négative

    # Cas de base : capacité nulle
    if s == 0:
        return 0  # Aucun bocal n'est nécessaire pour une capacité de 0

    # Cas de base : pas de bocaux restants et capacité non nulle
    if i == 0 and s >= 1:
        return float('inf')  # Impossible d'atteindre s > 0 sans bocaux

    # Cas général : relation de récurrence
    # On ne prend pas le bocal V[i]
    sans_bocal = m(s, i - 1, V)

    # On prend le bocal V[i]
    avec_bocal = m(s - V[i - 1], i, V) + 1

    return min(sans_bocal, avec_bocal)

# Algorithme II
def AlgoOptimise(S, k, V):
    """
    Algorithme pour calculer le nombre minimal de bocaux nécessaires pour atteindre une capacité S.

    :param S: Capacité totale à atteindre.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux (1-indexée, V[0] ignoré).
    :return: Le nombre minimal de bocaux nécessaires pour atteindre S, ou +∞ si impossible.
    """
    # Initialisation du tableau M avec +∞
    M = [[float('inf')] * (k + 1) for _ in range(S + 1)]

    # Initialisation des cas de base : m(0, i) = 0 pour tout i
    for i in range(k + 1):
        M[0][i] = 0

    # Remplissage du tableau selon la relation de récurrence
    for i in range(1, k + 1):  # Pour chaque type de bocal
        for s in range(1, S + 1):  # Pour chaque capacité de 1 à S
            M[s][i] = M[s][i - 1]  # Cas où on n'utilise pas le bocal V[i]
            if s >= V[i - 1]:
                M[s][i] = min(M[s][i], M[s - V[i - 1]][i] + 1)  # Cas où on utilise le bocal V[i]

    # Retour du résultat final
    return M[S][k]

def AlgoOptimiseV2(S, k, V):
    """
    Algorithme pour calculer le nombre minimal de bocaux nécessaires pour atteindre une capacité S,
    et pour fournir les types de bocaux utilisés dans la solution optimale.

    :param S: Capacité totale à atteindre.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux (1-indexée, V[0] ignoré).
    :return: A[S][k] - Liste des bocaux utilisés.
    """
    # Initialisation du tableau M avec +∞
    M = [[float('inf')] * (k + 1) for _ in range(S + 1)]
    # Initialisation du tableau A avec des listes vides
    A = [[[] for _ in range(k + 1)] for _ in range(S + 1)]

    # Initialisation des cas de base : m(0, i) = 0 pour tout i
    for i in range(k + 1):
        M[0][i] = 0  # Aucun bocal nécessaire pour une capacité de 0
        A[0][i] = []  # Aucun bocal utilisé

    # Remplissage du tableau selon la relation de récurrence
    for i in range(1, k + 1):  # Pour chaque type de bocal
        for s in range(1, S + 1):  # Pour chaque capacité de 1 à S
            # Cas où on ne prend pas le bocal V[i]
            M[s][i] = M[s][i - 1]
            A[s][i] = A[s][i - 1][:]  # Copier la solution précédente (sans le bocal V[i])

            # Cas où on prend le bocal V[i]
            if s >= V[i - 1]:
                if M[s][i] > M[s - V[i - 1]][i] + 1:
                    M[s][i] = M[s - V[i - 1]][i] + 1
                    A[s][i] = A[s - V[i - 1]][i][:]  # Copier la solution pour s - V[i]
                    A[s][i].append(V[i - 1])  # Ajouter le bocal V[i] à la solution

    # Retourner le résultat final
    return A[S][k]

def AlgoRetour(S, k, V):
    """
    Algorithme pour calculer la solution optimale pour remplir une capacité S en utilisant 
    un ensemble de bocaux de types V en fonction d'un tableau M (d'abord rempli avec programmation dynamique).

    :param S: Capacité totale à atteindre.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux (1-indexée, V[0] est ignoré).
    :return: Liste des bocaux utilisés dans la solution optimale.
    """
    # Étape 1 : Initialiser et remplir le tableau M
    M = [[float('inf')] * (k + 1) for _ in range(S + 1)]  # Tableau M rempli avec infini
    for i in range(k + 1):
        M[0][i] = 0  # m(0, i) = 0 pour toute i

    # Remplir le tableau M selon la relation de récurrence
    for i in range(1, k + 1):  # Pour chaque type de bocal
        for s in range(1, S + 1):  # Pour chaque capacité de 1 à S
            M[s][i] = M[s][i - 1]  # Cas où V[i] n'est pas utilisé
            if s >= V[i-1]:  # Si on peut utiliser le bocal V[i]
                M[s][i] = min(M[s][i - 1], M[s - V[i-1]][i] + 1)  # Cas général

    # Étape 2 : Processus de retour pour déterminer les bocaux utilisés
    A = []  # Liste vide pour stocker les bocaux utilisés
    s = S  # Capacité restante
    i = k  # Dernier type de bocal

    while s > 0 and i > 0:  # Tant que nous n'avons pas atteint la capacité 0
        # Si le bocal V[i] a été utilisé
        if s >= V[i-1] and M[s][i] == M[s - V[i-1]][i] + 1:
            A.append(V[i-1])  # Ajouter le bocal V[i] à la liste des bocaux utilisés
            s -= V[i-1]  # Réduire la capacité restante
        else:
            i -= 1  # Passer au type de bocal précédent

    # Retourner la liste des bocaux utilisés
    return A

# Algorithme III
def AlgoGlouton(S, k, V): # Complexité temporelle en O(S)
    """
    Algorithme glouton pour remplir la capacité S en utilisant les bocaux disponibles,
    en choisissant toujours le plus grand bocal possible.

    :param S: Capacité totale à remplir.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux triée (du plus petit au plus grand).
    :return: Le nombre total de bocaux utilisés, ou None si la capacité ne peut pas être atteinte.
    (et aussi le tableau des bocaux si on a besion)
    """
    index = k - 1  # Commencer avec le plus grand type de bocal
    nb_bocaux = 0  # Nombre total de bocaux utilisés
    tab_bocaux = [0] * k  # Tableau pour suivre le nombre de bocaux utilisés de chaque type

    # Tant que la capacité S n'est pas atteinte et qu'il reste des bocaux à considérer
    while S != 0 and index >= 0:
        # Si le bocal actuel peut être utilisé pour remplir la capacité restante
        if V[index] <= S:
            S -= V[index]  # Réduire la capacité S de la taille du bocal
            tab_bocaux[index] += 1  # Compter ce bocal comme utilisé
            nb_bocaux += 1  # Augmenter le nombre total de bocaux utilisés
        else:
            index -= 1  # Passer au bocal de taille plus petite

    # Si S est toujours supérieur à 0, cela signifie que la capacité ne peut pas être remplie
    if S > 0:
        return None
    # Retourne le nombre total de bocaux utilisés (nb_bocaux).
    # Si l'on souhaite également récupérer le tableau des bocaux utilisés (tab_bocaux),
    # il suffit de décommenter la ligne suivante pour obtenir la liste complète.
    return nb_bocaux  # ,tab_bocaux # (Retirer le '#' pour obtenir également le tableau des bocaux)

def AlgoGloutonV2(S, k, V): # # Complexité temporelle en O(K)
    """
    Version améliorée de l'algorithme glouton pour remplir la capacité S en utilisant les bocaux disponibles,
    en prenant autant de bocaux que possible du plus grand au plus petit type de bocal.

    :param S: Capacité totale à remplir.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux triée (du plus petit au plus grand).
    :return: Le nombre total de bocaux utilisés, ou None si la capacité ne peut pas être atteinte.
    (et aussi le tableau des bocaux si on a besion)
    """
    index = k - 1  # Commencer avec le plus grand type de bocal
    nb_bocaux = 0  # Nombre total de bocaux utilisés
    tab_bocaux = [0] * k  # Tableau pour suivre le nombre de bocaux utilisés de chaque type

    # Tant que la capacité S n'est pas atteinte et qu'il reste des bocaux à considérer
    while S != 0 and index >= 0:
        # Si le bocal actuel peut être utilisé pour remplir la capacité restante
        if V[index] <= S:
            # Calculer combien de bocaux de cette taille peuvent être utilisés
            nb_bocaux_temp = S // V[index]
            S -= nb_bocaux_temp * V[index]  # Réduire la capacité S
            tab_bocaux[index] += nb_bocaux_temp  # Mettre à jour le nombre de bocaux utilisés
            nb_bocaux += nb_bocaux_temp  # Augmenter le nombre total de bocaux utilisés
        index -= 1  # Passer au bocal de taille plus petite

    # Si S est toujours supérieur à 0, cela signifie que la capacité ne peut pas être remplie
    if S > 0:
        return None
    # Retourne le nombre total de bocaux utilisés (nb_bocaux).
    # Si l'on souhaite également récupérer le tableau des bocaux utilisés (tab_bocaux),
    # il suffit de décommenter la ligne suivante pour obtenir la liste complète.
    return nb_bocaux  # ,tab_bocaux # (Retirer le '#' pour obtenir également le tableau des bocaux)

def TestGloutonCompatible(k, V):
    """
    Vérifie si l'algorithme glouton est compatible avec les bocaux de capacités données.

    :param k: Le nombre de types de bocaux.
    :param V: Un tableau des capacités des bocaux de taille k.
    :return: True si l'algorithme glouton est compatible, False sinon.
    """
    # Vérifier si le nombre de types de bocaux est supérieur ou égal à 3
    if k >= 3:
        # Vérifier pour chaque capacité S de (V[3] + 2) à (V[k-1] + V[k] - 1)
        for S in range(V[2] + 2, V[k - 2] + V[k - 1]):
            # Pour chaque type de bocal
            for j in range(k):
                # Vérifier si V[j] est inférieur à S et si la condition de l'algorithme glouton est violée
                if V[j] < S and AlgoGlouton(S, k, V) > 1 + AlgoGlouton(S - V[j], k, V):
                    return False  # Si la condition est violée, retourner False
        return True  # Si aucune condition n'est violée, retourner True
    return True  # Si aucune condition n'est violée, retourner True

# Tests de fonctionnement
def test_algorithms(fichier):
    S, k, V = lire_entree(fichier)
    print(f"Capacité totale : {S}, Nombre de types de bocaux : {k}")
    print(f"Capacités des bocaux : {V}")

    # Test de m(s, i, V) pour un exemple
    print("Test de m(s, i, V) pour s = 151 et i = 3 : ", m(S, k, V))

    # Test de l'algorithme optimisé sans récupérer les bocaux utilisés
    print("Test de AlgoOptimise(S, k, V) : ", AlgoOptimise(S, k, V))

    # Test de l'algorithme optimisé avec récupération des bocaux utilisés
    print("Test de AlgoOptimiseV2(S, k, V) : ", AlgoOptimiseV2(S, k, V))

    # Test de l'algorithme avec retour pour récupérer les bocaux utilisés
    print("Test de AlgoRetour(S, k, V) : ", AlgoRetour(S, k, V))

    # Vérifier la compatibilité glouton
    print("Test de Glouton Compatible : ", TestGloutonCompatible(k, V))

    # Test de l'algorithme glouton (qui est en O(S))
    print("Test de AlgoGlouton(S, k, V) : ", AlgoGlouton(S, k, V))

    # Test de l'algorithme glouton version 2 (qui est en O(K))
    print("Test de AlgoGloutonV2(S, k, V) : ", AlgoGloutonV2(S, k, V))

# Lancer les tests sur le fichier 'donnee.txt'
test_algorithms('donnee.txt')

# Fonction pour comparer les performances en termes de vitesse des algorithmes AlgoOptimiseV2 et AlgoRetour.
def compare_AlgoOptimiseV2_avec_AlgoRetour(S_values, k, V):
    """
    Compare les performances en termes de vitesse des algorithmes AlgoOptimiseV2 et AlgoRetour.

    :param S_values: Liste des capacités totales (S) à tester.
    :param k: Nombre de types de bocaux disponibles.
    :param V: Liste des capacités des bocaux.
    :return: None (Affiche une comparaison graphique des temps d'exécution).
    """
    times_algo_opt = []  # Liste pour les temps d'exécution d'AlgoOptimiseV2
    times_algo_ret = []  # Liste pour les temps d'exécution d'AlgoRetour

    for S in S_values:
        # Mesurer le temps d'exécution pour AlgoOptimiseV2
        start_time = time.process_time()
        AlgoOptimiseV2(S, k, V)
        end_time = time.process_time()
        times_algo_opt.append(end_time - start_time)
        
        # Mesurer le temps d'exécution pour AlgoRetour
        start_time = time.process_time()
        AlgoRetour(S, k, V)
        end_time = time.process_time()
        times_algo_ret.append(end_time - start_time)

    # Création du graphique de comparaison
    plt.figure(figsize=(10, 6))
    plt.plot(S_values, times_algo_opt, label="AlgoOptimiseV2", marker="o", color="blue")
    plt.plot(S_values, times_algo_ret, label="AlgoRetour", marker="s", color="orange")

    # Personnalisation du graphique
    plt.xlabel("Capacité totale (S)")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.title("Comparaison des performances d'AlgoOptimiseV2 et AlgoRetour")
    plt.legend()
    plt.grid(True)
    plt.show()

# 3.2 Analyse de complexité expérimentale
# Question 12
def gen_sys_expo(d, k):
    """
    Génère une liste des puissances de d jusqu'à l'exposant k-1.
    
    Arguments :
    - d : base (entier ou flottant)
    - k : nombre de termes à générer (entier)
    
    Retourne :
    - Une liste des valeurs [d^0, d^1, ..., d^(k-1)].
    """
    return [d**i for i in range(k)]

def run_experiments(d, max_S, k):
    """
    Effectue des expériences pour tester les algorithmes I, II, III et III version 2.
    
    Arguments :
    - d : base pour générer le système exponentiel (entier)
    - max_S : capacité maximale S (entier)
    - k : nombre d'éléments dans le système (entier)
    
    Retourne :
    - S_values : liste des valeurs de S utilisées dans les tests
    - k_values : liste des valeurs de k utilisées dans les tests
    - times_algo_i : liste des temps d'exécution de l'algorithme I
    - times_algo_ii : liste des temps d'exécution de l'algorithme II
    - times_algo_iii : liste des temps d'exécution de l'algorithme III
    - times_algo_iiiV2 : liste des temps d'exécution de l'algorithme III version 2
    """
    times_algo_i = []
    times_algo_ii = []
    times_algo_iii = []
    times_algo_iiiV2 = []
    S_values = []
    k_values = []

    V = gen_sys_expo(d, k)

    for S in range(1, max_S + 1):
        # Test de l'algorithme I (récursif)
        start_time = time.process_time()
        m(S, k, V)  # Appel de la fonction récursive
        end_time = time.process_time()

        execution_time_i = end_time - start_time
        if execution_time_i > 60:  # Arrête si le temps d'exécution dépasse 1 minute
            print(f"L'algorithme I a dépassé la limite de temps pour S={S}, k={k}. Arrêt des tests.")
            break  # Arrête les tests pour cette combinaison de S et k
        times_algo_i.append(execution_time_i)

        # Test de l'algorithme II (programmation dynamique)
        start_time = time.process_time()
        AlgoOptimise(S, k, V)
        end_time = time.process_time()
        times_algo_ii.append(end_time - start_time)

        # Test de l'algorithme III (glouton)
        start_time = time.process_time()
        AlgoGlouton(S, k, V)
        end_time = time.process_time()
        times_algo_iii.append(end_time - start_time)

        # Test de la version 2 de l'algorithme glouton
        start_time = time.process_time()
        AlgoGloutonV2(S, k, V)
        end_time = time.process_time()
        times_algo_iiiV2.append(end_time - start_time)

        S_values.append(S)
        k_values.append(k)

    return S_values, k_values, times_algo_i, times_algo_ii, times_algo_iii, times_algo_iiiV2

def plot_results(S_values, k_values, times_algo_i, times_algo_ii, times_algo_iii, times_algo_iiiV2):
    """
    Trace les résultats des temps d'exécution des différents algorithmes en fonction de S.
    
    Arguments :
    - S_values : liste des capacités S testées
    - k_values : liste des valeurs de k utilisées
    - times_algo_i : liste des temps d'exécution de l'algorithme I
    - times_algo_ii : liste des temps d'exécution de l'algorithme II
    - times_algo_iii : liste des temps d'exécution de l'algorithme III
    - times_algo_iiiV2 : liste des temps d'exécution de l'algorithme III version 2
    
    Affiche un graphique des temps d'exécution.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(S_values, times_algo_i, label="Algo I (Récursif)", color='g')
    plt.plot(S_values, times_algo_ii, label="Algo II (Programmation Dynamique)", color='b')
    plt.plot(S_values, times_algo_iii, label="Algo III (Glouton) O(S)", color='r')
    plt.plot(S_values, times_algo_iiiV2, label="Algo III (Glouton) O(k)", color='y')

    plt.xlabel('Capacité S')
    plt.ylabel('Temps d\'exécution (en secondes)')
    plt.title('Temps d\'exécution en fonction de S')
    plt.legend()

    plt.grid(True)
    plt.show()

# 3.3 Utilisation de l'algorithme glouton
# Génération de systèmes de capacités
def gen_boc_rand(k, pmax):
    """
    Génère un système aléatoire avec k éléments et un maximum pmax.
    
    Arguments :
    - k : nombre d'éléments dans le système (entier)
    - pmax : valeur maximale des éléments (entier)
    
    Retourne :
    - Une liste triée contenant k éléments aléatoires (dont toujours 1).
    
    Exceptions :
    - ValueError si k est supérieur à pmax + 1 ou si la taille de l'échantillon dépasse la population.
    """
    if k > pmax + 1:
        raise ValueError("k plus grand que pmax + 1.")

    if k - 1 <= 0:
        return [1]

    if k - 1 > pmax - 1:
        raise ValueError("Taille de l'échantillon supérieure à la population.")

    # Les capacités des bocaux seront choisies uniformément entre 2 et une valeur maximum pmax
    boc_rand = random.sample(range(1, pmax + 1), k - 1)
    return sorted([1] + boc_rand)

# Question 13
def prop_sys_glo(max_k, pmax):
    """
    Calcule la proportion de systèmes compatibles avec l'algorithme glouton.
    
    Arguments :
    - max_k : nombre maximum d'éléments dans les systèmes (entier)
    - pmax : valeur maximale des éléments (entier)
    
    Retourne :
    - Proportion de systèmes compatibles (flottant).
    """
    nb_comp = 0
    for i in range(1, max_k + 1):
        V = gen_boc_rand(i, pmax)
        if TestGloutonCompatible(i, V):
            nb_comp += 1
    return nb_comp / max_k

# Question 14
def test_ecart(max_k, pmax, f):
    """
    Teste l'écart entre les résultats des algorithmes glouton et optimisé.
    
    Arguments :
    - max_k : nombre maximum d'éléments dans les systèmes (entier)
    - pmax : valeur maximale des éléments (entier)
    - f : facteur multiplicatif pour le test (entier ou flottant)
    
    Retourne :
    - Une liste contenant les écarts entre les solutions glouton et optimisée.
    """
    tab_ecart = []
    for i in range(1, max_k + 1):
        V = gen_boc_rand(i, pmax)
        if not TestGloutonCompatible(i, V):
            for S in range(pmax, (pmax * f) + 1):
                tab_ecart.append(AlgoGlouton(S, i, V) - AlgoOptimise(S, i, V))
    return tab_ecart


def plot_ecart_histogram(max_k, pmax, f):
    """
    Génère un histogramme des différences calculées par la fonction test_ecart,
    et affiche les valeurs moyennes, maximales et minimales.

    Arguments :
    - max_k : le nombre maximal de k pour la génération des données
    - pmax : la population maximale pour la génération des données
    - f : un facteur utilisé dans test_ecart pour déterminer la plage de S
    """
    # Appel à la fonction test_ecart pour obtenir les données
    data = test_ecart(max_k, pmax, f)
    
    # Tracer l'histogramme
    plt.hist(data, bins=20)
    plt.title("Histogramme des différences")
    plt.xlabel("Différence")
    plt.ylabel("Fréquence")

    # Calcul des valeurs moyennes, maximales et minimales
    mean_value = statistics.mean(data)
    max_value = max(data)
    min_value = min(data)

    # Ajouter des lignes pour la moyenne, le maximum et le minimum
    plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(max_value, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(min_value, color='y', linestyle='dashed', linewidth=1)

    # Ajouter des annotations pour la moyenne, le maximum et le minimum
    min_ylim, max_ylim = plt.ylim()
    plt.text(mean_value, max_ylim * 0.9, f'Moyenne: {mean_value:.2f}', color='r')
    plt.text(max_value, max_ylim * 0.8, f'Max: {max_value:.2f}', color='g')
    plt.text(min_value, max_ylim * 0.97, f'Min: {min_value:.2f}', color='y')

    # Afficher le graphique
    plt.show()

# Nous pouvons tester tous les fcontiosn du projet par le menu suivant:
def menu():
    while True:
        print("\n*** Menu Principal ***")
        print("1. Tester l'algorithme récursif (m)")
        print("2. Tester l'algorithme optimisé (AlgoOptimise)")
        print("3. Tester l'algorithme optimisé avec bocaux utilisés (AlgoOptimiseV2)")
        print("4. Tester l'algorithme avec retour (AlgoRetour)")
        print("5. Tester l'algorithme glouton (AlgoGlouton)")
        print("6. Tester l'algorithme glouton version2 (AlgoGloutonV2)")
        print("7. Vérifier la compatibilité glouton (TestGloutonCompatible)")
        print("8. Générer un système exponentiel et comparer les algorithmes")
        print("9. Charger les données d'un fichier et tester tous les algorithmes")
        print("10. Comparer les performances d'AlgoOptimise2 et AlgoRetour")
        print("11. Générer un système de bocaux aléatoire (gen_boc_rand)")
        print("12. Calculer la proportion de systèmes compatibles glouton (prop_sys_glo)")
        print("13. Teste l'écart entre les résultats des algorithmes glouton et optimisé en génèrant un histogramme(plot_ecart_histogram)")
        print("14. Quitter")
        choix = input("Choisissez une option : ")

        if choix == '1':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Résultat de m(S, k, V) : {m(S, k, V)}")

        elif choix == '2':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Résultat de AlgoOptimise(S, k, V) : {AlgoOptimise(S, k, V)}")

        elif choix == '3':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Bocaux utilisés (AlgoOptimiseV2) : {AlgoOptimiseV2(S, k, V)}")

        elif choix == '4':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Bocaux utilisés (AlgoRetour) : {AlgoRetour(S, k, V)}")

        elif choix == '5':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Nombre de bocaux utilisés (AlgoGlouton) : {AlgoGlouton(S, k, V)}")

        elif choix == '6':
            S = int(input("Entrez la capacité totale (S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Nombre de bocaux utilisés (AlgoGloutonV2) : {AlgoGloutonV2(S, k, V)}")

        elif choix == '7':
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
            print(f"Compatibilité glouton : {TestGloutonCompatible(k, V)}")

        elif choix == '8':
            d = int(input("Entrez la base exponentielle (d) : "))
            max_S = int(input("Entrez la capacité maximale (max_S) : "))
            k = int(input("Entrez le nombre de types de bocaux (k) : "))
            print("Lancement des expériences...")
            S_values, k_values, times_algo_i, times_algo_ii, times_algo_iii, times_algo_iiiV2 = run_experiments(d, max_S, k)
            plot_results(S_values, k_values, times_algo_i, times_algo_ii, times_algo_iii, times_algo_iiiV2)
        
        elif choix == '9':
            fichier = input("Entrez le chemin du fichier d'entrée : ")
            try:
                test_algorithms(fichier)
            except Exception as e:
                print(f"Erreur lors du test des algorithmes avec le fichier '{fichier}' : {e}")
        
        elif choix == "10":
            try:
                S_values = list(map(int, input("Entrez une liste de capacités (S) séparées par des espaces : ").split()))
                k = int(input("Entrez le nombre de types de bocaux (k) : "))
                V = list(map(int, input("Entrez les capacités des bocaux séparées par des espaces : ").split()))
                print("\nLancement de la comparaison des algorithmes...")
                compare_AlgoOptimiseV2_avec_AlgoRetour(S_values, k, V)
            except ValueError:
                print("Entrée invalide. Veuillez réessayer.")
        
        elif choix == '11':
            try:
                k = int(input("Entrez le nombre de types de bocaux (k) : "))
                pmax = int(input("Entrez la capacité maximale (pmax) : "))
                print(f"Système de bocaux généré : {gen_boc_rand(k, pmax)}")
            except ValueError as e:
                print(f"Erreur : {e}")

        elif choix == '12':
            try:
                max_k = int(input("Entrez le nombre maximal de types de bocaux (max_k) : "))
                pmax = int(input("Entrez la capacité maximale (pmax) : "))
                proportion = prop_sys_glo(max_k, pmax)
                print(f"Proportion de systèmes compatibles glouton : {proportion:.2f}")
            except ValueError as e:
                print(f"Erreur : {e}")
        
        elif choix == '13':
            try:
                max_k = int(input("Entrez le nombre maximal de types de bocaux (max_k) : "))
                pmax = int(input("Entrez la capacité maximale (pmax) : "))
                f = int(input("Entrez le facteur multiplicatif (f) : "))
                plot_ecart_histogram(max_k, pmax, f)
            except ValueError as e:
                print(f"Erreur : {e}")

        elif choix == '14':
            print("Au revoir!")
            break

        else:
            print("Option invalide, veuillez réessayer.")

# Exécutez le menu
menu()
