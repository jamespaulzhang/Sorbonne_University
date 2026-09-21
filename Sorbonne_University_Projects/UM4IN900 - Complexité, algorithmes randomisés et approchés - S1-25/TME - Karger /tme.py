import copy
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class MultiGraphAdjMatrix:
    """Graphe multiple représenté par une matrice d'adjacence"""

    def __init__(self, n_vertices):
        """
        Initialise un graphe multiple avec n sommets

        Args:
            n_vertices: Nombre de sommets
        """
        self.n = n_vertices
        # Initialise la matrice d'adjacence, les éléments représentent le nombre d'arêtes entre les sommets
        self.adj_matrix = [[0 for _ in range(n_vertices)] for _ in range(n_vertices)]
        # Mappage des étiquettes des sommets, utilisé pour suivre quels sommets originaux ont été fusionnés
        self.vertex_labels = [{i} for i in range(n_vertices)]
        # Si le sommet est actif (certains sommets peuvent être fusionnés lors de la contraction)
        self.active = [True for _ in range(n_vertices)]
        self.active_count = n_vertices

    def add_edge(self, u, v):
        """
        Ajoute une arête (u, v)

        Args:
            u, v: Indices des sommets (de 0 à n-1)
        """
        if u == v:
            return  # Pas de boucle autorisée

        self.adj_matrix[u][v] += 1
        self.adj_matrix[v][u] += 1  # Graphe non orienté, symétrique

    def add_weighted_edge(self, u, v, weight):
        """
        Ajoute une arête (u, v) avec un poids donné

        Args:
            u, v: Indices des sommets
            weight: Poids de l'arête (représente le nombre d'arêtes parallèles)
        """
        if u == v:
            return

        self.adj_matrix[u][v] += weight
        self.adj_matrix[v][u] += weight

    def get_edge_count(self, u, v):
        """
        Récupère le nombre d'arêtes entre les sommets u et v
        """
        return self.adj_matrix[u][v]

    def contract_edge(self, edge):
        """
        Contracte une arête spécifiée, complexité temporelle O(n)

        Args:
            edge: Arête à contracter, format (u, v)

        Returns:
            Nouveau graphe (modifié sur place, mais retourné pour plus de clarté)
        """
        u, v = edge

        # Si u ou v est inactif, ou si u == v, ne rien faire
        if not self.active[u] or not self.active[v] or u == v:
            return self

        # Fusionner le sommet v dans le sommet u
        for i in range(self.n):
            if self.active[i] and i != u and i != v:
                # Ajouter le nombre d'arêtes entre v et i au nombre d'arêtes entre u et i
                self.adj_matrix[u][i] += self.adj_matrix[v][i]
                self.adj_matrix[i][u] += self.adj_matrix[v][i]

                # Supprimer les arêtes entre v et i
                self.adj_matrix[v][i] = 0
                self.adj_matrix[i][v] = 0

        # Fusionner les étiquettes des sommets (enregistrer quels sommets originaux ont été fusionnés)
        self.vertex_labels[u].update(self.vertex_labels[v])

        # Supprimer les arêtes entre u et v (maintenant une boucle, à supprimer)
        self.adj_matrix[u][v] = 0
        self.adj_matrix[v][u] = 0

        # Marquer v comme inactif (fusionné dans u)
        self.active[v] = False
        self.active_count -= 1

        # Supprimer la boucle sur v
        self.adj_matrix[v][v] = 0

        return self

    def get_total_edges(self):
        """
        Récupère le nombre total d'arêtes dans le graphe (en comptant les arêtes parallèles)

        Returns:
            Nombre total d'arêtes
        """
        total = 0
        for i in range(self.n):
            if self.active[i]:
                for j in range(i + 1, self.n):
                    if self.active[j]:
                        total += self.adj_matrix[i][j]
        return total

    def get_all_edges(self):
        """
        Récupère la liste de toutes les arêtes du graphe (en comptant les arêtes parallèles)

        Returns:
            Liste des arêtes, chaque arête au format (u, v, poids)
        """
        edges = []
        for i in range(self.n):
            if self.active[i]:
                for j in range(i + 1, self.n):
                    if self.active[j] and self.adj_matrix[i][j] > 0:
                        edges.append((i, j, self.adj_matrix[i][j]))
        return edges

    def get_vertices(self):
        """
        Récupère la liste des sommets actifs
        """
        return [i for i in range(self.n) if self.active[i]]

    def get_vertex_mapping(self):
        """
        Récupère la relation de mappage des sommets fusionnés
        """
        return self.vertex_labels

    def copy(self):
        """
        Crée une copie profonde du graphe
        """
        new_graph = MultiGraphAdjMatrix(self.n)
        # Copier la matrice d'adjacence
        new_graph.adj_matrix = [row[:] for row in self.adj_matrix]
        # Copier les étiquettes des sommets
        new_graph.vertex_labels = [label.copy() for label in self.vertex_labels]
        # Copier l'état actif
        new_graph.active = self.active[:]
        new_graph.active_count = self.active_count
        return new_graph
    
    def choose_random_edge_reservoir(self):
        """
        Sélectionne une arête aléatoire uniformément en utilisant l'algorithme du réservoir (reservoir sampling)
        Complexité temporelle: O(n²) - un seul parcours de la matrice d'adjacence
        Complexité spatiale: O(1)
        
        Returns:
            Une arête (u, v) choisie aléatoirement, ou None si le graphe n'a pas d'arêtes
        """
        n = self.n
        arête_choisie = None  # L'arête actuellement sélectionnée
        k = 0  # Nombre d'arêtes vues jusqu'à présent (compte les arêtes parallèles)
        
        # Parcourir seulement la partie triangulaire supérieure de la matrice pour éviter les doublons
        for i in range(n):
            if not self.active[i]:
                continue
                
            for j in range(i + 1, n):
                if not self.active[j]:
                    continue
                    
                # Obtenir le nombre d'arêtes entre i et j (pour les multigraphes)
                nombre_arêtes = self.adj_matrix[i][j]
                
                if nombre_arêtes > 0:
                    # Pour chaque arête parallèle
                    for _ in range(nombre_arêtes):
                        k += 1
                        # Reservoir sampling: la k-ème arête a une probabilité 1/k d'être sélectionnée
                        if random.random() < 1.0 / k:
                            arête_choisie = (i, j)
        
        return arête_choisie
    
    def choose_random_edge_weighted(self):
        """
        Sélectionne une arête aléatoire uniformément en utilisant une version pondérée du reservoir sampling
        Cette version est optimisée pour les multigraphes avec de nombreuses arêtes parallèles
        
        Returns:
            Une arête (u, v) choisie aléatoirement, ou None si le graphe n'a pas d'arêtes
        """
        n = self.n
        arête_choisie = None
        arêtes_vues = 0  # Compte cumulatif des arêtes (pondéré)
        
        for i in range(n):
            if not self.active[i]:
                continue
                
            for j in range(i + 1, n):
                if not self.active[j]:
                    continue
                    
                poids = self.adj_matrix[i][j]  # Nombre d'arêtes entre i et j
                
                if poids > 0:
                    # Reservoir sampling pondéré: une arête avec poids w a une probabilité w/(total+w) d'être sélectionnée
                    arêtes_vues += poids
                    if random.random() < poids / arêtes_vues:
                        arête_choisie = (i, j)
        
        return arête_choisie
    
    def choose_random_edge_two_pass(self):
        """
        Sélectionne une arête aléatoire en deux passes: d'abord compter toutes les arêtes, puis en choisir une
        Complexité temporelle: O(n²) - deux parcours de la matrice
        Complexité spatiale: O(1)
        
        Returns:
            Une arête (u, v) choisie aléatoirement, ou None si le graphe n'a pas d'arêtes
        """
        n = self.n
        total_arêtes = self.get_total_edges()
        
        if total_arêtes == 0:
            return None
        
        # Choisir un indice d'arête aléatoire
        indice_arête = random.randint(1, total_arêtes)
        
        # Deuxième passe: trouver l'arête correspondante
        compteur_actuel = 0
        for i in range(n):
            if not self.active[i]:
                continue
                
            for j in range(i + 1, n):
                if not self.active[j]:
                    continue
                    
                nombre_arêtes = self.adj_matrix[i][j]
                if nombre_arêtes > 0:
                    compteur_actuel += nombre_arêtes
                    if compteur_actuel >= indice_arête:
                        return (i, j)
        
        return None

    def __str__(self):
        """
        Représentation en chaîne de caractères du graphe
        """
        result = f"Graphe multiple (nombre de sommets actifs : {self.active_count})\n"
        result += "Matrice d'adjacence (seulement les sommets actifs) :\n"

        active_vertices = self.get_vertices()
        if not active_vertices:
            return result + "Graphe vide"

        # Afficher les étiquettes des colonnes
        result += "   " + " ".join(f"{v:3}" for v in active_vertices) + "\n"

        # Afficher le contenu de la matrice
        for i in active_vertices:
            result += f"{i:2} "
            for j in active_vertices:
                result += f"{self.adj_matrix[i][j]:3} "
            result += "\n"

        return result


def create_complete_graph(n):
    """
    Crée un graphe complet avec n sommets

    Args:
        n: Nombre de sommets

    Returns:
        Représentation par matrice d'adjacence du graphe complet
    """
    graph = MultiGraphAdjMatrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j)
    return graph


def create_cycle_graph(n):
    """
    Crée un graphe cyclique avec n sommets

    Args:
        n: Nombre de sommets

    Returns:
        Représentation par matrice d'adjacence du graphe cyclique
    """
    graph = MultiGraphAdjMatrix(n)
    for i in range(n):
        graph.add_edge(i, (i + 1) % n)
    return graph


def create_complete_bipartite_graph(k):
    """
    Crée un graphe biparti complet K_{k,k}, avec un total de 2k sommets

    Args:
        k: Nombre de sommets dans chaque partition

    Returns:
        Représentation par matrice d'adjacence du graphe biparti complet
    """
    n = 2 * k
    graph = MultiGraphAdjMatrix(n)
    # Première partition : sommets 0 à k-1
    # Deuxième partition : sommets k à 2k-1
    for i in range(k):
        for j in range(k, n):
            graph.add_edge(i, j)
    return graph


def create_random_graph(n, p):
    """
    Crée un graphe aléatoire où chaque arête existe avec une probabilité p

    Args:
        n: Nombre de sommets
        p: Probabilité qu'une arête existe

    Returns:
        Représentation par matrice d'adjacence du graphe aléatoire
    """
    graph = MultiGraphAdjMatrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                graph.add_edge(i, j)
    return graph


def measure_contraction_time(graph, edge, repetitions=10):
    """
    Mesure le temps moyen pour contracter une arête

    Args:
        graph: Objet graphe
        edge: Arête à contracter
        repetitions: Nombre de répétitions pour calculer la moyenne

    Returns:
        Temps moyen de contraction (en secondes)
    """
    total_time = 0

    for _ in range(repetitions):
        # Créer une copie profonde du graphe pour éviter que les contractions multiples n'affectent les tests suivants
        test_graph = graph.copy()

        start_time = time.perf_counter()
        test_graph.contract_edge(edge)
        end_time = time.perf_counter()

        total_time += (end_time - start_time)

    return total_time / repetitions


def test_contraction_complexity():
    """
    Teste la complexité de l'opération de contraction sur différentes familles de graphes
    """
    results = {
        'cycle': {'n_values': [], 'times': [], 'complexity': []},
        'complete': {'n_values': [], 'times': [], 'complexity': []},
        'bipartite': {'n_values': [], 'times': [], 'complexity': []},
        'random': {'n_values': [], 'times': [], 'complexity': []}
    }

    # Paramètres de test
    n_values = list(range(10, 151, 10))  # De 10 à 150, pas de 10
    p = 0.3  # Probabilité des arêtes pour les graphes aléatoires
    k_values = [n // 2 for n in n_values]  # Valeurs de k pour les graphes bipartis complets

    print("Début des tests de complexité temporelle pour la contraction des arêtes...")
    print("=" * 60)

    # 1. Tester les graphes cycliques
    print("\n1. Test des graphes cycliques (Cycle Graph) :")
    for n in n_values:
        graph = create_cycle_graph(n)
        # Sélectionner une arête aléatoire pour la contraction
        edges = graph.get_all_edges()
        if edges:
            edge = (edges[0][0], edges[0][1])  # Prendre la première arête
            avg_time = measure_contraction_time(graph, edge, repetitions=5)

            results['cycle']['n_values'].append(n)
            results['cycle']['times'].append(avg_time)
            results['cycle']['complexity'].append(avg_time / n)  # Facteur constant pour O(n)

            print(f"  n={n:3d}, temps={avg_time:.8f}s, facteur constant={avg_time/n:.8f}")

    # 2. Tester les graphes complets
    print("\n2. Test des graphes complets (Complete Graph) :")
    for n in n_values[:10]:  # Les graphes complets sont trop grands, tester seulement les 10 premières valeurs
        graph = create_complete_graph(n)
        # Sélectionner une arête aléatoire pour la contraction
        edges = graph.get_all_edges()
        if edges:
            edge = (edges[0][0], edges[0][1])
            avg_time = measure_contraction_time(graph, edge, repetitions=3)

            results['complete']['n_values'].append(n)
            results['complete']['times'].append(avg_time)
            results['complete']['complexity'].append(avg_time / n)

            print(f"  n={n:3d}, temps={avg_time:.8f}s, facteur constant={avg_time/n:.8f}")

    # 3. Tester les graphes bipartis complets
    print("\n3. Test des graphes bipartis complets (Complete Bipartite Graph) :")
    for n, k in zip(n_values, k_values):
        if k > 0:
            graph = create_complete_bipartite_graph(k)
            # Sélectionner une arête aléatoire pour la contraction
            edges = graph.get_all_edges()
            if edges:
                edge = (edges[0][0], edges[0][1])
                avg_time = measure_contraction_time(graph, edge, repetitions=5)

                results['bipartite']['n_values'].append(2*k)
                results['bipartite']['times'].append(avg_time)
                results['bipartite']['complexity'].append(avg_time / (2*k))

                print(f"  n={2*k:3d} (k={k:3d}), temps={avg_time:.8f}s, facteur constant={avg_time/(2*k):.8f}")

    # 4. Tester les graphes aléatoires
    print("\n4. Test des graphes aléatoires (Random Graph, p=0.3) :")
    for n in n_values:
        graph = create_random_graph(n, p)
        # Sélectionner une arête aléatoire pour la contraction
        edges = graph.get_all_edges()
        if edges:
            # Sélectionner une arête aléatoire
            random_edge = random.choice(edges)
            edge = (random_edge[0], random_edge[1])
            avg_time = measure_contraction_time(graph, edge, repetitions=5)

            results['random']['n_values'].append(n)
            results['random']['times'].append(avg_time)
            results['random']['complexity'].append(avg_time / n)

            print(f"  n={n:3d}, temps={avg_time:.8f}s, facteur constant={avg_time/n:.8f}")

    return results


def analyze_complexity(results):
    """
    Analyse les résultats expérimentaux et calcule la tendance de complexité
    """
    print("\n" + "="*60)
    print("Analyse de la complexité")
    print("="*60)

    for graph_type, data in results.items():
        if data['n_values'] and data['times']:
            n_values = np.array(data['n_values'])
            times = np.array(data['times'])

            # Régression linéaire pour T = a*n + b
            coeffs = np.polyfit(n_values, times, 1)
            a, b = coeffs

            print(f"\nGraphe de type {graph_type.upper()} :")
            print(f"  Résultat de l'ajustement : T(n) = {a:.10f} * n + {b:.10f}")
            print(f"  Complexité théorique : O(n)")
            print(f"  Plage du facteur constant : {min(data['complexity']):.10f} à {max(data['complexity']):.10f}")

            # Calcul du R²
            predictions = a * n_values + b
            ss_res = np.sum((times - predictions) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            print(f"  Coefficient de détermination R² = {r_squared:.6f}")

            if r_squared > 0.95:
                print(f"  ✓ Bon ajustement linéaire, conforme à la complexité O(n)")
            elif r_squared > 0.8:
                print(f"  ~ Ajustement linéaire acceptable, globalement conforme à la complexité O(n)")
            else:
                print(f"  ✗ Mauvais ajustement linéaire, peut ne pas être conforme à la complexité O(n)")


def plot_results(results):
    """
    Trace les graphiques des résultats expérimentaux
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sous-graphe 1 : Temps vs Nombre de sommets
    ax1 = axes[0, 0]
    for graph_type, data in results.items():
        if data['n_values'] and data['times']:
            ax1.plot(data['n_values'], data['times'], 'o-', label=graph_type)
    ax1.set_xlabel('Nombre de sommets (n)')
    ax1.set_ylabel('Temps de contraction (s)')
    ax1.set_title('Temps de contraction des arêtes vs Nombre de sommets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sous-graphe 2 : Facteur constant vs Nombre de sommets
    ax2 = axes[0, 1]
    for graph_type, data in results.items():
        if data['n_values'] and data['complexity']:
            ax2.plot(data['n_values'], data['complexity'], 'o-', label=graph_type)
    ax2.set_xlabel('Nombre de sommets (n)')
    ax2.set_ylabel('Facteur constant (temps/n)')
    ax2.set_title('Facteur constant de la complexité temporelle vs Nombre de sommets')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Sous-graphe 3 : Graphique log-log
    ax3 = axes[1, 0]
    for graph_type, data in results.items():
        if data['n_values'] and data['times']:
            ax3.loglog(data['n_values'], data['times'], 'o-', label=graph_type)

    # Ajouter une ligne de référence y = x (complexité linéaire)
    x_ref = np.array([min(n for n in data['n_values'] if data['n_values']),
                      max(n for n in data['n_values'] if data['n_values'])])
    y_ref = x_ref * np.mean([np.mean(data['complexity']) for data in results.values() if data['complexity']])
    ax3.loglog(x_ref, y_ref, 'k--', label='Ligne de référence O(n)')

    ax3.set_xlabel('Nombre de sommets (n)')
    ax3.set_ylabel('Temps de contraction (s)')
    ax3.set_title('Graphique log-log : Temps de contraction des arêtes vs Nombre de sommets')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Sous-graphe 4 : Comparaison des temps pour chaque type de graphe (n fixe)
    ax4 = axes[1, 1]
    # Trouver une valeur commune de n pour la comparaison
    common_n = 50
    times_at_n = {}

    for graph_type, data in results.items():
        if common_n in data['n_values']:
            idx = data['n_values'].index(common_n)
            times_at_n[graph_type] = data['times'][idx]

    if times_at_n:
        ax4.bar(times_at_n.keys(), times_at_n.values())
        ax4.set_xlabel('Type de graphe')
        ax4.set_ylabel('Temps de contraction (s)')
        ax4.set_title(f'Temps de contraction pour différents types de graphes avec n={common_n}')

        # Ajouter des étiquettes de valeur sur le graphique à barres
        for i, (graph_type, time_val) in enumerate(times_at_n.items()):
            ax4.text(i, time_val, f'{time_val:.8f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('contraction_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def detailed_analysis_per_graph():
    """
    Analyse détaillée pour chaque type de graphe
    """
    print("\n" + "="*60)
    print("Analyse détaillée des caractéristiques de chaque type de graphe")
    print("="*60)

    # Analyser des graphes de différentes tailles
    test_sizes = [20, 50, 100]

    for n in test_sizes:
        print(f"\nNombre de sommets n = {n}:")
        print("-" * 40)

        # Graphe cyclique
        cycle = create_cycle_graph(n)
        cycle_edges = cycle.get_all_edges()
        if cycle_edges:
            edge = (cycle_edges[0][0], cycle_edges[0][1])
            time_cycle = measure_contraction_time(cycle, edge)
            print(f"  Graphe cyclique : nombre d'arêtes={len(cycle_edges)}, temps de contraction={time_cycle:.8f}s")

        # Graphe complet (testé uniquement pour les petites valeurs de n)
        if n <= 30:
            complete = create_complete_graph(n)
            complete_edges = complete.get_all_edges()
            if complete_edges:
                edge = (complete_edges[0][0], complete_edges[0][1])
                time_complete = measure_contraction_time(complete, edge)
                print(f"  Graphe complet : nombre d'arêtes={len(complete_edges)}, temps de contraction={time_complete:.8f}s")

        # Graphe biparti complet
        if n % 2 == 0:
            k = n / 2
            bipartite = create_complete_bipartite_graph(k)
            bipartite_edges = bipartite.get_all_edges()
            if bipartite_edges:
                edge = (bipartite_edges[0][0], bipartite_edges[0][1])
                time_bipartite = measure_contraction_time(bipartite, edge)
                print(f"  Graphe biparti complet : nombre d'arêtes={len(bipartite_edges)}, temps de contraction={time_bipartite:.8f}s")

        # Graphe aléatoire
        random_g = create_random_graph(n, 0.3)
        random_edges = random_g.get_all_edges()
        if random_edges:
            random_edge = random.choice(random_edges)
            edge = (random_edge[0], random_edge[1])
            time_random = measure_contraction_time(random_g, edge)
            print(f"  Graphe aléatoire (p=0.3) : nombre d'arêtes≈{len(random_edges)}, temps de contraction={time_random:.8f}s")


def test_random_edge_selection():
    """
    Teste les algorithmes de sélection aléatoire d'arêtes
    """
    print("\n" + "="*60)
    print("Tests des algorithmes de sélection aléatoire d'arêtes")
    print("="*60)
    
    # Créer un graphe de test
    n = 6
    graph = MultiGraphAdjMatrix(n)
    
    # Ajouter des arêtes
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(0, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 0)
    
    # Ajouter des arêtes parallèles pour tester les multigraphes
    graph.add_edge(0, 1)  # Arête parallèle
    graph.add_edge(0, 1)  # Une autre arête parallèle
    
    print(f"Nombre total d'arêtes (avec arêtes parallèles) : {graph.get_total_edges()}")
    print("Toutes les arêtes du graphe :")
    for arête in graph.get_all_edges():
        print(f"  ({arête[0]}, {arête[1]}) - poids : {arête[2]}")
    
    # Méthodes à tester
    méthodes = [
        ("Reservoir sampling standard", lambda g: g.choose_random_edge_reservoir()),
        ("Reservoir sampling pondéré", lambda g: g.choose_random_edge_weighted()),
        ("Deux passes", lambda g: g.choose_random_edge_two_pass()),
    ]
    
    # Nombre d'itérations pour les tests statistiques
    itérations = 10000
    
    for nom_méthode, fonction in méthodes:
        print(f"\nTest de la méthode : {nom_méthode}")
        
        # Statistiques de sélection
        statistiques = {}
        
        # Exécuter plusieurs fois pour vérifier la distribution
        for _ in range(itérations):
            arête = fonction(graph)
            if arête is not None:
                clé_arête = f"{arête[0]}-{arête[1]}"
                statistiques[clé_arête] = statistiques.get(clé_arête, 0) + 1
        
        # Calculer les probabilités théoriques et observées
        total_arêtes = graph.get_total_edges()
        liste_arêtes = graph.get_all_edges()
        
        print("Statistiques de sélection :")
        for arête in liste_arêtes:
            clé_arête = f"{arête[0]}-{arête[1]}"
            compte = statistiques.get(clé_arête, 0)
            prob_observée = compte / itérations
            prob_théorique = arête[2] / total_arêtes  # Poids de l'arête divisé par le nombre total d'arêtes
            
            print(f"  Arête ({arête[0]}, {arête[1]}) - poids : {arête[2]}")
            print(f"    Nombre de sélections : {compte} / {itérations}")
            print(f"    Probabilité observée : {prob_observée:.4f}")
            print(f"    Probabilité théorique : {prob_théorique:.4f}")
            print(f"    Différence : {abs(prob_observée - prob_théorique):.4f}")


def mesure_temps_sélection_arête():
    """
    Mesure le temps d'exécution des différents algorithmes de sélection d'arête
    """
    print("\n" + "="*60)
    print("Mesure des temps d'exécution pour la sélection d'arêtes")
    print("="*60)
    
    # Tailles de graphes à tester
    valeurs_n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    méthodes = [
        ("Reservoir standard", lambda g: g.choose_random_edge_reservoir()),
        ("Reservoir pondéré", lambda g: g.choose_random_edge_weighted()),
        ("Deux passes", lambda g: g.choose_random_edge_two_pass()),
    ]
    
    résultats = {nom: [] for nom, _ in méthodes}
    
    for n in valeurs_n:
        print(f"\nTest avec n = {n}")
        
        # Créer un graphe aléatoire
        graph = create_random_graph(n, 0.3)
        
        for nom_méthode, fonction in méthodes:
            # Mesurer le temps moyen sur plusieurs répétitions
            répétitions = 100
            temps_total = 0
            
            for _ in range(répétitions):
                début = time.perf_counter()
                arête = fonction(graph)
                fin = time.perf_counter()
                temps_total += (fin - début)
            
            temps_moyen = temps_total / répétitions
            résultats[nom_méthode].append(temps_moyen)
            
            print(f"  {nom_méthode} : {temps_moyen:.8f}s")


if __name__ == "__main__":
    # Menu principal pour exécuter les différents tests
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL - IMPLÉMENTATION DE L'ALGORITHME DE KARGER")
        print("="*60)
        print("1. Tester la complexité de la contraction d'arêtes")
        print("2. Tester les algorithmes de sélection aléatoire d'arêtes")
        print("3. Mesurer les temps de sélection d'arêtes")
        print("4. Quitter")
        
        choix = input("\nVotre choix (1-6) : ")
        
        if choix == "1":
            print("\n" + "="*60)
            print("TEST DE LA COMPLEXITÉ DE CONTRACTION")
            print("="*60)
            résultats = test_contraction_complexity()
            analyze_complexity(résultats)
            plot_results(résultats)
            
        elif choix == "2":
            print("\n" + "="*60)
            print("TEST DES ALGORITHMES DE SÉLECTION ALÉATOIRE")
            print("="*60)
            test_random_edge_selection()
            
        elif choix == "3":
            print("\n" + "="*60)
            print("MESURE DES TEMPS DE SÉLECTION")
            print("="*60)
            mesure_temps_sélection_arête()
            
        elif choix == "4":
            print("\nAu revoir !")
            break
            
        else:
            print("Choix invalide. Veuillez sélectionner une option entre 1 et 6.")