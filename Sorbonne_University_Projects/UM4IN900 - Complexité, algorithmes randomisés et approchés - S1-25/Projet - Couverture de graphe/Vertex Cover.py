import random
import time
import matplotlib.pyplot as plt
import math
import itertools

def read_graph(filename):
    """
    Lit un graphe à partir d'un fichier texte.
    
    Args:
        filename: Chemin vers le fichier contenant la définition du graphe
        
    Returns:
        dict: Dictionnaire représentant la liste d'adjacence du graphe
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx_sommets = lines.index("Sommets")
    idx_aretes = lines.index("Aretes")
    
    sommets = [int(x) for x in lines[idx_sommets + 1 : lines.index("Nombre d aretes")]]
    adj = {s: [] for s in sommets}
    
    for line in lines[idx_aretes + 1 :]:
        u, v = map(int, line.split())
        adj[u].append(v)
        adj[v].append(u)
    
    return adj

class Graph:
    """Classe représentant un graphe non orienté avec des listes d'adjacence."""
    
    def __init__(self, adj):
        """
        Initialise le graphe avec une liste d'adjacence.
        
        Args:
            adj: Dictionnaire des listes d'adjacence
        """
        self.adj = adj

    def sommets(self):
        """Retourne la liste des sommets du graphe."""
        return list(self.adj.keys())

    def aretes(self):
        """
        Retourne la liste des arêtes du graphe.
        
        Returns:
            list: Liste de tuples (u, v) représentant les arêtes
        """
        edges = set()
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in edges:
                    edges.add((u, v))
        return list(edges)

    def degree(self, v):
        """Retourne le degré du sommet v."""
        return len(self.adj[v])
    
    def degrees_dict(self):
        """
        Retourne un dictionnaire des degrés de tous les sommets.
        
        Returns:
            dict: Dictionnaire {sommet: degré}
        """
        return {u: len(neigh) for u, neigh in self.adj.items()}

    def degrees_list(self):
        """
        Retourne une liste des degrés de tous les sommets.
        
        Returns:
            list: Liste des degrés dans l'ordre des sommets
        """
        verts = self.sommets()
        return [self.degree(v) for v in verts]

    def max_degree_vertex(self, return_all=True):
        """
        Trouve le(s) sommet(s) de degré maximal.
        
        Args:
            return_all: Si True, retourne tous les sommets de degré maximal
                       Si False, retourne un seul sommet arbitraire
                       
        Returns:
            tuple: (liste des sommets, degré maximal) ou (sommet, degré maximal)
        """
        verts = self.sommets()
        if not verts:
            return ([], 0) if return_all else (None, 0)
        
        degs = self.degrees_dict()
        max_deg = max(degs.values())
        max_vertices = [v for v in verts if degs.get(v, 0) == max_deg]

        if return_all:
            return (max_vertices, max_deg)
        else:
            return (max_vertices[0], max_deg)
    
    def copy(self):
        """Retourne une copie profonde du graphe."""
        return Graph({u: list(neigh) for u, neigh in self.adj.items()})
    
    def remove_vertex(self, v):
        """
        Retire un sommet du graphe et retourne un nouveau graphe.
        
        Args:
            v: Sommet à retirer
            
        Returns:
            Graph: Nouveau graphe sans le sommet v
        """
        if v not in self.adj:
            return self.copy()
        
        new_adj = {u: [x for x in neigh if x != v] for u, neigh in self.adj.items() if u != v}
        return Graph(new_adj)
    
    def remove_vertices(self, vertices):
        """
        Retire plusieurs sommets du graphe et retourne un nouveau graphe.
        
        Args:
            vertices: Ensemble ou liste de sommets à retirer
            
        Returns:
            Graph: Nouveau graphe sans les sommets spécifiés
        """
        to_remove = set(vertices)

        if not to_remove:
            return self.copy()

        new_adj = {}
        for u, neigh in self.adj.items():
            if u in to_remove:
                continue
            filtered = [w for w in neigh if w not in to_remove]
            new_adj[u] = filtered

        return Graph(new_adj)
    
    def remove_vertices_inplace(self, vertices):
        """
        Retire plusieurs sommets du graphe en modifiant le graphe actuel.
        
        Args:
            vertices: Ensemble ou liste de sommets à retirer
        """
        to_remove = set(vertices)
        for v in list(to_remove):
            if v in self.adj:
                del self.adj[v]
        for u, neigh in self.adj.items():
            self.adj[u] = [w for w in neigh if w not in to_remove]

    def est_couverture_valide(self, C):
        """
        Vérifie si un ensemble de sommets est une couverture valide.
        
        Args:
            C: Ensemble de sommets à vérifier
            
        Returns:
            bool: True si C couvre toutes les arêtes, False sinon
        """
        for u, v in self.aretes():
            if u not in C and v not in C:
                return False
        return True
    
    def algo_couplage(self):
        """
        Algorithme de couplage pour le problème de couverture de sommets.
        
        Returns:
            set: Ensemble de sommets formant une couverture
        """
        edges = self.aretes()

        C = set()
        for (u, v) in edges:
            if (u not in C) and (v not in C):
                C.add(u)
                C.add(v)
        return C
    
    def algo_glouton(self):
        """
        Algorithme glouton pour le problème de couverture de sommets.
        
        Returns:
            set: Ensemble de sommets formant une couverture
        """
        Gc = self.copy()
        C = set()
        while True:
            edges = Gc.aretes()
            if not edges:
                break
            v, _ = Gc.max_degree_vertex(return_all=False)
            C.add(v)
            Gc.remove_vertices_inplace({v})
        return C
    
    def branchement_simple(self):
        """
        Algorithme de branchement simple pour la couverture de sommets.
        Évite de stocker des copies complètes du graphe.
        
        Returns:
            tuple: (meilleure couverture, nombre de nœuds générés)
        """
        best_C = set(self.sommets())
        
        # Pile : chaque élément est (arêtes_restantes, solution_courante)
        stack = []
        initial_edges = self.aretes()
        stack.append((initial_edges, set()))
        
        nodes_generated = 0  # Compteur de nœuds générés

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # S'il n'y a plus d'arêtes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Choisir une arête pour le branchement
            u, v = remaining_edges[0]

            # Branche 1 : choisir le sommet u
            new_edges1 = [(x, y) for (x, y) in remaining_edges if x != u and y != u]
            new_solution1 = current_solution | {u}
            stack.append((new_edges1, new_solution1))

            # Branche 2 : choisir le sommet v
            new_edges2 = [(x, y) for (x, y) in remaining_edges if x != v and y != v]
            new_solution2 = current_solution | {v}
            stack.append((new_edges2, new_solution2))

        return best_C, nodes_generated
    
    def _borne_inf(self, remaining_edges):
        """
        Calcule une borne inférieure pour le nombre de sommets nécessaires
        pour couvrir les arêtes restantes.
        """
        if not remaining_edges:
            return 0
            
        # Construire le sous-graphe
        sub_vertices = set()
        for (x, y) in remaining_edges:
            sub_vertices.add(x)
            sub_vertices.add(y)
        sub_adj = {v: [] for v in sub_vertices}
        for (x, y) in remaining_edges:
            sub_adj[x].append(y)
            sub_adj[y].append(x)

        # Paramètres du sous-graphe
        n_sub = len(sub_vertices)
        m_sub = len(remaining_edges)
        
        # Calcul du degré max Δ dans le sous-graphe
        if n_sub == 0:
            delta_sub = 0
        else:
            delta_sub = max((len(neigh) for neigh in sub_adj.values()), default=0)

        # b1 = ceil(m / Δ) si Δ > 0, sinon 0
        if delta_sub > 0:
            b1 = math.ceil(m_sub / delta_sub)
        else:
            b1 = 0

        # b2 = taille du couplage maximal (approché gloutonnement)
        matched = set()
        matching_size = 0
        for (a, b) in remaining_edges:
            if a not in matched and b not in matched:
                matched.add(a)
                matched.add(b)
                matching_size += 1
        b2 = matching_size

        # b3 = borne basée sur la formule quadratique
        if n_sub > 0:
            inner = (2 * n_sub - 1) ** 2 - 8 * m_sub
            inner = max(inner, 0.0)
            b3_val = (2 * n_sub - 1 - math.sqrt(inner)) / 2.0
            b3 = math.ceil(b3_val)
            if b3 < 0:
                b3 = 0
        else:
            b3 = 0

        # Retourne le maximum des bornes
        return max(b1, b2, b3)
    
    def branchement_couplage_avec_borne(self):
        """
        Algorithme de branchement simple pour la couverture de sommets.
        À chaque nœud, on calcule :
          - une solution réalisable par l'algorithme de couplage sur le sous-graphe
          - des bornes inférieures b1, b2, b3 et b = max(b1,b2,b3)
        On utilise lower_bound = len(current_solution) + b pour le pruning.

        Returns:
            tuple: (meilleure couverture, nombre de nœuds générés)
        """
        best_C = set(self.sommets())

        # Pile : chaque élément est (arêtes_restantes, solution_courante)
        stack = []
        initial_edges = self.aretes()
        stack.append((initial_edges, set()))

        nodes_generated = 0  # Compteur de nœuds générés

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Si plus d'arêtes, on a une solution réelle (courante)
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = set(current_solution)
                continue

            # --- Construire le sous-graphe induit par remaining_edges ---
            sub_vertices = set()
            for (x, y) in remaining_edges:
                sub_vertices.add(x)
                sub_vertices.add(y)
            sub_adj = {v: [] for v in sub_vertices}
            for (x, y) in remaining_edges:
                sub_adj[x].append(y)
                sub_adj[y].append(x)

            # Paramètres du sous-graphe
            n_sub = len(sub_vertices)
            m_sub = len(remaining_edges)
            # Calcul du degré max Δ dans le sous-graphe
            if n_sub == 0:
                delta_sub = 0
            else:
                delta_sub = max((len(neigh) for neigh in sub_adj.values()), default=0)

            # --- Calcul d'un matching (greedy) sur remaining_edges to get b2 = |M| ---
            # On va construire un matching M_edges en marquant les sommets utilisés
            matched = set()
            matching_size = 0
            for (a, b) in remaining_edges:
                if a not in matched and b not in matched:
                    matched.add(a)
                    matched.add(b)
                    matching_size += 1
            b2 = matching_size

            # --- Calcul de b1, b3 ---
            # b1 = ceil(m / Δ) si Δ > 0, sinon 0
            if delta_sub > 0:
                b1 = math.ceil(m_sub / delta_sub)
            else:
                b1 = 0

            # b3 = ceil( (2n-1 - sqrt((2n-1)^2 - 8m)) / 2 )
            # protéger l'intérieur de la racine contre une petite négativité numérique
            if n_sub > 0:
                inner = (2 * n_sub - 1) ** 2 - 8 * m_sub
                inner = max(inner, 0.0)
                b3_val = (2 * n_sub - 1 - math.sqrt(inner)) / 2.0
                b3 = math.ceil(b3_val)
                if b3 < 0:
                    b3 = 0
            else:
                b3 = 0

            # borne b = max(b1, b2, b3)
            b_lower = max(b1, b2, b3)

            # --- Calculer une solution réalisable via algo_couplage sur le sous-graphe ---
            subG = Graph(sub_adj)
            C_couplage_sub = subG.algo_couplage()  # ensemble de sommets couvrant toutes les remaining_edges

            # Solution réalisable en ce nœud: current_solution ∪ C_couplage_sub
            feasible_solution = current_solution | C_couplage_sub
            # Si cette solution réalisable est meilleure que best_C, on la conserve
            if len(feasible_solution) < len(best_C):
                best_C = set(feasible_solution)

            # Borne inférieure combinée : current_solution + b_lower
            lower_bound = len(current_solution) + b_lower

            # Si la borne n'améliore pas best_C, on coupe ce nœud (pruning)
            if lower_bound >= len(best_C):
                continue

            # Choisir une arête pour le branchement (heuristique simple: la première)
            u, v = remaining_edges[0]

            # Branche 1 : choisir le sommet u
            new_edges1 = [(x, y) for (x, y) in remaining_edges if x != u and y != u]
            new_solution1 = current_solution | {u}
            stack.append((new_edges1, new_solution1))

            # Branche 2 : choisir le sommet v
            new_edges2 = [(x, y) for (x, y) in remaining_edges if x != v and y != v]
            new_solution2 = current_solution | {v}
            stack.append((new_edges2, new_solution2))

        return best_C, nodes_generated
    
    def branchement_avec_glouton_seulement(self):
        """
        Branchement utilisant l'algorithme glouton pour générer des solutions réalisables
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Solution par algorithme glouton pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                # Utiliser l'algorithme glouton
                couverture_glouton = Graph(temp_adj).algo_glouton()
                candidate = current_solution.union(couverture_glouton)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_glouton_avec_borne(self):
        """
        Branchement utilisant l'algorithme glouton pour générer des solutions réalisables
        ET les bornes inférieures pour le pruning
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Calcul de la borne inférieure
            LB = self._borne_inf(remaining_edges)
            
            # Élagage par borne inférieure
            if len(current_solution) + LB >= len(best_C):
                continue

            # Solution par algorithme glouton pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                # Utiliser l'algorithme glouton
                couverture_glouton = Graph(temp_adj).algo_glouton()
                candidate = current_solution.union(couverture_glouton)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated

    def branchement_avec_bornes_seulement(self):
        """
        Branchement utilisant seulement les bornes inférieures (sans solutions réalisables)
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Calcul de la borne inférieure
            LB = self._borne_inf(remaining_edges)
            
            # Élagage par borne inférieure
            if len(current_solution) + LB >= len(best_C):
                continue

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated

    def branchement_avec_couplage_seulement(self):
        """
        Branchement utilisant seulement le couplage (borne inférieure triviale = 0)
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Solution par couplage pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                couverture_couplage = Graph(temp_adj).algo_couplage()
                candidate = current_solution.union(couverture_couplage)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_ameliore_v1(self):
        """
        Branchement amélioré version 1 : dans la deuxième branche, exclure les cas déjà traités
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage
            if len(current_solution) >= len(best_C):
                continue

            # Si pas d'arêtes restantes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Choisir une arête pour le branchement
            u, v = remaining_edges[0]

            # Branche 1 : choisir le sommet u
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))

            # Branche 2 : choisir le sommet v, et ne pas choisir u (donc doit choisir tous les voisins de u)
            # Obtenir tous les voisins de u (sauf v, car v est déjà choisi)
            neighbors_u = set()
            for x, y in remaining_edges:
                if x == u and y != v:
                    neighbors_u.add(y)
                elif y == u and x != v:
                    neighbors_u.add(x)
            
            # Dans la deuxième branche, choisir v et tous les voisins de u
            sol2 = current_solution | {v} | neighbors_u
            
            # Supprimer toutes les arêtes liées à u, v et les voisins de u
            vertices_to_remove = {u, v} | neighbors_u
            edges2 = [(x, y) for x, y in remaining_edges 
                    if x not in vertices_to_remove and y not in vertices_to_remove]
            
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_ameliore_v2(self):
        """
        Branchement amélioré version 2 : choisir le sommet de degré maximal pour le branchement
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage
            if len(current_solution) >= len(best_C):
                continue

            # Si pas d'arêtes restantes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Calculer le degré de chaque sommet dans le graphe actuel
            degree = {}
            for u, v in remaining_edges:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1

            # Trouver l'arête avec le sommet de degré maximal
            max_degree = -1
            best_edge = None
            for u, v in remaining_edges:
                current_max = max(degree.get(u, 0), degree.get(v, 0))
                if current_max > max_degree:
                    max_degree = current_max
                    best_edge = (u, v)

            u, v = best_edge
            
            # S'assurer que u est le sommet avec le degré le plus élevé
            if degree.get(v, 0) > degree.get(u, 0):
                u, v = v, u

            # Branche 1 : choisir le sommet de degré maximal u
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))

            # Branche 2 : choisir v, et ne pas choisir u (donc doit choisir tous les voisins de u)
            neighbors_u = set()
            for x, y in remaining_edges:
                if x == u and y != v:
                    neighbors_u.add(y)
                elif y == u and x != v:
                    neighbors_u.add(x)
            
            sol2 = current_solution | {v} | neighbors_u
            vertices_to_remove = {u, v} | neighbors_u
            edges2 = [(x, y) for x, y in remaining_edges 
                    if x not in vertices_to_remove and y not in vertices_to_remove]
            
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_ameliore_v3(self):
        """
        Branchement amélioré version 3 : traitement spécial des sommets de degré 1
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage
            if len(current_solution) >= len(best_C):
                continue

            # Traitement spécial des sommets de degré 1
            modified = True
            while modified and remaining_edges:
                modified = False
                # Calculer les degrés des sommets dans le graphe restant
                degree = {}
                for u, v in remaining_edges:
                    degree[u] = degree.get(u, 0) + 1
                    degree[v] = degree.get(v, 0) + 1
                
                # Chercher les sommets de degré 1
                for vertex in list(degree.keys()):
                    if degree.get(vertex, 0) == 1:
                        # Trouver l'unique voisin de ce sommet
                        neighbor = None
                        for u, v in remaining_edges:
                            if u == vertex:
                                neighbor = v
                                break
                            elif v == vertex:
                                neighbor = u
                                break
                        
                        if neighbor is not None:
                            # Ajouter le voisin à la solution (pas le sommet de degré 1)
                            current_solution.add(neighbor)
                            # Supprimer toutes les arêtes incidentes au voisin
                            remaining_edges = [(x, y) for x, y in remaining_edges 
                                            if x != neighbor and y != neighbor]
                            modified = True
                            break
            
            # Si pas d'arêtes restantes après traitement, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Calculer le degré de chaque sommet dans le graphe actuel
            degree = {}
            for u, v in remaining_edges:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1

            # Trouver l'arête avec le sommet de degré maximal
            max_degree = -1
            best_edge = None
            for u, v in remaining_edges:
                current_max = max(degree.get(u, 0), degree.get(v, 0))
                if current_max > max_degree:
                    max_degree = current_max
                    best_edge = (u, v)

            u, v = best_edge
            
            # S'assurer que u est le sommet avec le degré le plus élevé
            if degree.get(v, 0) > degree.get(u, 0):
                u, v = v, u

            # Branche 1 : choisir le sommet de degré maximal u
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))

            # Branche 2 : choisir v, et ne pas choisir u (donc doit choisir tous les voisins de u)
            neighbors_u = set()
            for x, y in remaining_edges:
                if x == u and y != v:
                    neighbors_u.add(y)
                elif y == u and x != v:
                    neighbors_u.add(x)
            
            sol2 = current_solution | {v} | neighbors_u
            vertices_to_remove = {u, v} | neighbors_u
            edges2 = [(x, y) for x, y in remaining_edges 
                    if x not in vertices_to_remove and y not in vertices_to_remove]
            
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def heuristique_degre_ponderee(self, num_trials=50):
        """
        Heuristique de degré pondéré : choisit les sommets avec probabilité proportionnelle au degré
        
        Args:
            num_trials: Nombre d'essais
            
        Returns:
            set: Meilleure couverture trouvée
        """
        best_cover = set(self.sommets())
        best_size = len(best_cover)
        
        for _ in range(num_trials):
            cover = set()
            G_temp = self.copy()
            
            while G_temp.aretes():
                # Calculer les degrés
                degrees = G_temp.degrees_dict()
                total_degree = sum(degrees.values())
                
                if total_degree == 0:
                    break
                    
                # Choisir un sommet avec probabilité proportionnelle au degré
                rand_val = random.random()
                cumulative_prob = 0
                
                for vertex, degree in degrees.items():
                    prob = degree / total_degree
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        # Ajouter ce sommet à la couverture
                        cover.add(vertex)
                        G_temp.remove_vertices_inplace([vertex])
                        break
            
            if len(cover) < best_size and self.est_couverture_valide(cover):
                best_cover = cover
                best_size = len(cover)
        
        return best_cover

    def heuristique_recherche_locale(self, initial_cover=None, max_iter=1000):
        """
        Heuristique de recherche locale : améliore une solution initiale par recherche locale
        
        Args:
            initial_cover: Couverture initiale (si None, utilise l'algorithme glouton)
            max_iter: Nombre maximum d'itérations
            
        Returns:
            set: Meilleure couverture trouvée
        """
        if initial_cover is None:
            current_cover = self.algo_glouton()
        else:
            current_cover = set(initial_cover)
        
        best_cover = set(current_cover)
        best_size = len(best_cover)
        
        vertices = self.sommets()
        
        for iteration in range(max_iter):
            improved = False
            
            # Essayer de retirer un sommet
            for v in list(current_cover):
                if v in current_cover:
                    temp_cover = current_cover - {v}
                    if self.est_couverture_valide(temp_cover):
                        current_cover = temp_cover
                        improved = True
                        break
            
            # Si pas d'amélioration, essayer d'échanger deux sommets
            if not improved:
                for v_in in list(current_cover):
                    for v_out in vertices:
                        if v_out not in current_cover:
                            temp_cover = (current_cover - {v_in}) | {v_out}
                            if self.est_couverture_valide(temp_cover) and len(temp_cover) <= len(current_cover):
                                current_cover = temp_cover
                                improved = True
                                break
                    if improved:
                        break
            
            # Mettre à jour la meilleure solution
            if len(current_cover) < best_size:
                best_cover = set(current_cover)
                best_size = len(best_cover)
            
            # Critère d'arrêt si pas d'amélioration après plusieurs itérations
            if iteration > 50 and not improved:
                break
        
        return best_cover

    def heuristique_hybride(self):
        """
        Heuristique hybride : combine plusieurs méthodes
        
        Returns:
            set: Meilleure couverture trouvée
        """
        # Générer plusieurs solutions avec différentes méthodes
        solutions = []
        
        # 1. Algorithme glouton
        solutions.append(self.algo_glouton())
        
        # 2. Algorithme de couplage
        solutions.append(self.algo_couplage())
        
        # 3. Heuristique aléatoire
        if hasattr(self, 'heuristique_aleatoire'):
            solutions.append(self.heuristique_aleatoire(num_trials=20))
        else:
            solutions.append(self.algo_glouton())
        
        # 4. Heuristique de degré pondéré
        solutions.append(self.heuristique_degre_ponderee(num_trials=20))
        
        # 5. Recherche locale sur chaque solution
        improved_solutions = []
        for sol in solutions:
            improved = self.heuristique_recherche_locale(sol, max_iter=200)
            improved_solutions.append(improved)
        
        # Retourner la meilleure solution
        best_solution = min(improved_solutions, key=len)
        return best_solution

    def heuristique_aleatoire(self, num_trials=100):
        """
        Heuristique aléatoire : génère plusieurs couvertures aléatoires et garde la meilleure
        
        Args:
            num_trials: Nombre d'essais aléatoires
            
        Returns:
            set: Meilleure couverture trouvée
        """
        best_cover = set(self.sommets())
        best_size = len(best_cover)
        
        for _ in range(num_trials):
            # Générer une couverture aléatoire
            cover = set()
            remaining_edges = self.aretes()
            
            while remaining_edges:
                # Choisir une arête aléatoire
                u, v = random.choice(remaining_edges)
                # Choisir aléatoirement un des deux sommets
                if random.random() < 0.5:
                    cover.add(u)
                    remaining_edges = [(x, y) for x, y in remaining_edges if x != u and y != u]
                else:
                    cover.add(v)
                    remaining_edges = [(x, y) for x, y in remaining_edges if x != v and y != v]
            
            if len(cover) < best_size and self.est_couverture_valide(cover):
                best_cover = cover
                best_size = len(cover)
        
        return best_cover
   
    def __repr__(self):
        """Représentation textuelle du graphe."""
        return f"Graph({self.adj})"

def generate_random_graph(n: int, p: float):
    """
    Génère un graphe aléatoire selon le modèle G(n, p).
    
    Args:
        n: Nombre de sommets
        p: Probabilité qu'une arête existe entre deux sommets
        
    Returns:
        Graph: Graphe aléatoire généré
        
    Raises:
        ValueError: Si n <= 0 ou p n'est pas dans (0, 1)
    """
    if n <= 0:
        raise ValueError("n doit être > 0")
    if not (0 < p < 1):
        raise ValueError("p doit être dans l'intervalle (0, 1)")

    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].append(j)
                adj[j].append(i)

    return Graph(adj)

def mesure_algo(G, algo_name):
    """
    Mesure le temps d'exécution d'un algorithme sur un graphe.
    
    Args:
        G: Graphe d'entrée
        algo_name: Nom de l'algorithme ("glouton" ou "couplage")
        
    Returns:
        tuple: (temps d'exécution, taille de la couverture)
    """
    start = time.time()
    if algo_name == "glouton":
        C = G.algo_glouton()
    elif algo_name == "couplage":
        C = G.algo_couplage()
    else:
        raise ValueError("Algo inconnu")
    t = time.time() - start
    return t, len(C)

def mesure_temps_et_qualite(algo_name, n, p, num_instances=10):
    """
    Retourne le temps moyen et la taille moyenne de la couverture sur plusieurs instances.
    
    Args:
        algo_name: Nom de l'algorithme
        n: Nombre de sommets
        p: Probabilité des arêtes
        num_instances: Nombre d'instances à tester
        
    Returns:
        tuple: (temps moyen, taille moyenne de couverture)
    """
    temps_list = []
    taille_list = []
    for _ in range(num_instances):
        G = generate_random_graph(n, p)
        t, taille = mesure_algo(G, algo_name)
        temps_list.append(t)
        taille_list.append(taille)
    return sum(temps_list)/num_instances, sum(taille_list)/num_instances

def trouver_Nmax(algo_name, p=0.3, seuil_sec=3):
    """
    Trouve la taille Nmax pour laquelle l'algorithme s'exécute en moins de seuil_sec secondes.
    
    Args:
        algo_name: Nom de l'algorithme
        p: Probabilité des arêtes
        seuil_sec: Seuil de temps en secondes
        
    Returns:
        int: Taille Nmax estimée
    """
    n = 1
    while True:
        t, _ = mesure_temps_et_qualite(algo_name, n, p, num_instances=3)
        if t > seuil_sec:
            return n
        n += 1

def tests_algos(p=0.3, num_instances=10):
    """
    Teste et compare les algorithmes glouton et de couplage.
    
    Args:
        p: Probabilité des arêtes
        num_instances: Nombre d'instances par test
        
    Returns:
        tuple: Résultats des tests
    """
    # 1. Identifier Nmax
    Nmax_glouton = trouver_Nmax("glouton", p)
    Nmax_couplage = trouver_Nmax("couplage", p)
    Nmax = min(Nmax_glouton, Nmax_couplage)  # prendre la taille commune pour comparaison
    print(f"Nmax estimé = {Nmax}")

    # 2. Définir les 10 points de test
    ns = [int(Nmax * i / 10) for i in range(1, 11)]

    temps_glouton, taille_glouton = [], []
    temps_couplage, taille_couplage = [], []

    # 3. Mesurer temps moyen et couverture moyenne
    for n in ns:
        t_g, s_g = mesure_temps_et_qualite("glouton", n, p, num_instances)
        t_c, s_c = mesure_temps_et_qualite("couplage", n, p, num_instances)
        temps_glouton.append(t_g)
        taille_glouton.append(s_g)
        temps_couplage.append(t_c)
        taille_couplage.append(s_c)

    # 4. Tracer les courbes temps (échelle log-log)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(
        [math.log2(n) for n in ns],
        [math.log2(t) if t > 0 else 0 for t in temps_glouton],
        'r-o', label='Glouton'
    )
    plt.plot(
        [math.log2(n) for n in ns],
        [math.log2(t) if t > 0 else 0 for t in temps_couplage],
        'g-o', label='Couplage'
    )
    plt.xlabel("log₂(n)")
    plt.ylabel("log₂(Temps moyen en s)")
    plt.title(f"Échelle log-log (p={p})")
    plt.grid(True)
    plt.legend()

    # 5. Tracer les courbes qualité (taille de couverture)
    plt.subplot(1,2,2)
    plt.plot(ns, taille_glouton, 'r-o', label='Glouton')
    plt.plot(ns, taille_couplage, 'g-o', label='Couplage')
    plt.xlabel("Nombre de sommets n")
    plt.ylabel("Taille moyenne couverture")
    plt.title("Qualité des solutions")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ns, temps_glouton, temps_couplage, taille_glouton, taille_couplage

def bruteforce_vertex_cover(adj):
    """
    Algorithme de force brute pour trouver la couverture de sommets optimale.
    Retourne la première couverture optimale trouvée.
    """
    # Obtenir la liste triée des sommets
    V = sorted(adj.keys())
    m = len(V)
    
    # Construire la liste des arêtes (sans doublons)
    edges = [(u,v) for u in adj for v in adj[u] if u < v]
    
    # Initialiser les variables pour suivre la meilleure solution
    best = None
    best_size = float('inf')
    found_at_size = False
    
    # Explorer les sous-ensembles par taille croissante (de 0 à m)
    for r in range(0, m+1):
        found_at_size = False
        
        # Générer tous les sous-ensembles de taille r
        for subset in itertools.combinations(V, r):
            S = set(subset)
            
            # Vérifier si S est une couverture valide
            valid = all(u in S or v in S for (u,v) in edges)
            
            # Si c'est une couverture valide et meilleure ou égale à la courante
            if valid and len(S) <= best_size:
                best = S
                best_size = len(S)
                found_at_size = True
                # On continue à chercher à cette taille pour trouver toutes les solutions optimales
                # Mais on garde seulement la première trouvée
        
        # Élagage: si on a trouvé une solution de taille r, inutile de chercher des tailles > r
        if found_at_size:
            break
            
    return best

def bruteforce_vertex_cover_all(adj):
    """
    Retourne toutes les couvertures optimales (pour debug)
    """
    V = sorted(adj.keys())
    edges = [(u,v) for u in adj for v in adj[u] if u < v]
    
    best_solutions = []
    best_size = float('inf')
    
    for r in range(0, len(V)+1):
        current_size_found = False
        
        for subset in itertools.combinations(V, r):
            S = set(subset)
            valid = all(u in S or v in S for (u,v) in edges)
            
            if valid:
                if len(S) < best_size:
                    # Nouvelle meilleure taille
                    best_solutions = [S]
                    best_size = len(S)
                    current_size_found = True
                elif len(S) == best_size:
                    # Même taille optimale
                    best_solutions.append(S)
                    current_size_found = True
        
        if current_size_found:
            break
            
    return best_solutions

def tester_strategies_branchement(n_values, p_values_labels, num_instances=3, max_time_par_instance=30, strategies_to_run=None):
    """
    Teste différentes stratégies de branchement sur plusieurs instances
    """
    if strategies_to_run is None:
        strategies_to_run = ['simple', 'couplage_borne', 'glouton', 'bornes', 'couplage', 'glouton_borne']

    all_results = {}
    
    # Mapping des noms de stratégies vers les méthodes
    strategy_methods = {
        'simple': 'branchement_simple',
        'couplage_borne': 'branchement_couplage_avec_borne',
        'glouton': 'branchement_avec_glouton_seulement', 
        'bornes': 'branchement_avec_bornes_seulement',
        'couplage': 'branchement_avec_couplage_seulement',
        'glouton_borne': 'branchement_glouton_avec_borne'
    }

    for n in n_values:
        for p_label in p_values_labels:
            if p_label == '1/sqrt':
                p = 1.0 / math.sqrt(n)
            else:
                p = float(p_label)
            
            key = (n, p)
            all_results[key] = {s: [] for s in strategies_to_run}

            for inst in range(num_instances):
                G = generate_random_graph(n, p)
                print(f"n={n}, p={p:.4f}, inst={inst+1}/{num_instances}")

                for strategy in strategies_to_run:
                    method_name = strategy_methods[strategy]
                    method = getattr(G, method_name)
                    
                    t0 = time.time()
                    try:
                        result = method()
                        t1 = time.time()
                        
                        if isinstance(result, tuple):
                            coverage, nodes = result
                        else:
                            coverage = result
                            nodes = None
                        
                        valid = G.est_couverture_valide(coverage)
                        timeout = (t1 - t0) > max_time_par_instance
                        
                        all_results[key][strategy].append({
                            'time': t1 - t0,
                            'nodes': nodes,
                            'size': len(coverage),
                            'valid': valid,
                            'timeout': timeout
                        })
                        
                        if timeout:
                            print(f"  {strategy}: TIMEOUT")
                        else:
                            print(f"  {strategy}: OK - {len(coverage)} sommets, {t1-t0:.2f}s")
                        
                    except Exception as e:
                        print(f"  {strategy}: ERREUR - {e}")
                        all_results[key][strategy].append({
                            'time': None, 'nodes': None, 'size': None, 
                            'valid': False, 'error': str(e), 'timeout': False
                        })

    return all_results

def mesurer_branchement_instance_return_nodes(n, p, methode='simple', max_time_par_instance=60):
    """
    Génère une instance G(n,p), exécute la méthode spécifiée et renvoie les résultats.
    
    Args:
        n: Nombre de sommets
        p: Probabilité des arêtes
        methode: 'simple', 'couplage_borne' ou 'test'
        max_time_par_instance: Temps maximum par instance (secondes)
        
    Returns:
        tuple: (temps, taille_couverture, valide_flag, timeout_flag, nodes_generated)
    """
    G = generate_random_graph(n, p)
    t0 = time.time()
    
    if methode == 'simple':
        result = G.branchement_simple()
    elif methode == 'couplage_borne':
        result = G.branchement_couplage_avec_borne()
    elif methode == 'test':
        result = G.branchement_avec_glouton_seulement()
    else:
        raise ValueError(f"Méthode inconnue : {methode}")

    t = time.time() - t0

    if isinstance(result, tuple) and len(result) == 2:
        C, nodes_generated = result
    else:
        C = result
        nodes_generated = None

    valide = G.est_couverture_valide(C)
    timeout_flag = (t > max_time_par_instance)
    return t, len(C), valide, timeout_flag, nodes_generated

def tester_branchement_sur_une_valeur_p_complet(n_values, p_values_labels, num_instances=3, max_time_par_instance=30, methods_to_run=None):
    """
    Pour chaque n in n_values et chaque p in p_values_labels (float ou '1/sqrt'),
    génère num_instances graphes aléatoires et exécute uniquement les méthodes spécifiées.
    methods_to_run: liste parmi ['simple', 'couplage_borne', 'test'], par défaut toutes.
    Retourne un dict structuré similaire à avant.
    """
    if methods_to_run is None:
        methods_to_run = ['simple', 'couplage_borne', 'test']

    all_results = {}
    for n in n_values:
        for p_label in p_values_labels:
            if p_label == '1/sqrt':
                p = 1.0 / math.sqrt(n)
            else:
                p = float(p_label)
            key = (n, p)
            all_results[key] = {m: [] for m in methods_to_run}

            for inst in range(num_instances):
                G = generate_random_graph(n, p)

                if 'simple' in methods_to_run:
                    t0 = time.time()
                    try:
                        res_s = G.branchement_simple()
                        t1 = time.time()
                        cov_s, nodes_s = (res_s if isinstance(res_s, tuple) else (res_s, None))
                        valid_s = G.est_couverture_valide(cov_s)
                        all_results[key]['simple'].append({
                            'time': t1 - t0, 'nodes': nodes_s, 'size': len(cov_s), 'valid': valid_s
                        })
                    except Exception as e:
                        all_results[key]['simple'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})

                if 'couplage_borne' in methods_to_run:
                    t0 = time.time()
                    try:
                        res_c = G.branchement_couplage_avec_borne()
                        t1 = time.time()
                        cov_c, nodes_c = (res_c if isinstance(res_c, tuple) else (res_c, None))
                        valid_c = G.est_couverture_valide(cov_c)
                        all_results[key]['couplage_borne'].append({
                            'time': t1 - t0, 'nodes': nodes_c, 'size': len(cov_c), 'valid': valid_c
                        })
                    except Exception as e:
                        all_results[key]['couplage_borne'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})  # 修改这里

                if 'test' in methods_to_run:
                    t0 = time.time()
                    try:
                        res_t = G.branchement_avec_glouton_seulement()
                        t1 = time.time()
                        cov_t, nodes_t = (res_t if isinstance(res_t, tuple) else (res_t, None))
                        valid_t = G.est_couverture_valide(cov_t)
                        all_results[key]['test'].append({
                            'time': t1 - t0, 'nodes': nodes_t, 'size': len(cov_t), 'valid': valid_t
                        })
                    except Exception as e:
                        all_results[key]['test'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})

                print(f"n={n}, p={p:.4f}, inst={inst+1}/{num_instances} done.")

    return all_results

def tracer_comparaison_strategies_complet(all_results, strategies_to_plot=None, title_suffix=""):
    """
    Graphique complet avec toutes les stratégies et tous les p sur les mêmes axes
    """
    keys = list(all_results.keys())
    if not keys:
        print("Aucune donnée à tracer!")
        return
        
    ns = sorted({k[0] for k in keys})
    
    if strategies_to_plot is None:
        example_key = keys[0]
        strategies_to_plot = list(all_results[example_key].keys())

    # Couleurs et styles
    strategy_colors = {
        'simple': 'blue', 'couplage_borne': 'green', 'glouton': 'red',
        'bornes': 'orange', 'couplage': 'purple', 'glouton_borne': 'brown'
    }
    
    strategy_markers = {
        'simple': 'o', 'couplage_borne': 's', 'glouton': '^',
        'bornes': 'D', 'couplage': 'v', 'glouton_borne': '*'
    }
    
    strategy_names = {
        'simple': 'Simple', 'couplage_borne': 'Couplage+Borne', 
        'glouton': 'Glouton', 'bornes': 'Borne seule',
        'couplage': 'Couplage seul', 'glouton_borne': 'Glouton+Borne'
    }

    p_linestyles = {
        0.1: '-', 0.3: '--', 0.5: ':', 
        0.3536: '-.', 0.3162: '-.', 0.2887: '-.', 0.2673: '-.', 0.2500: '-.'
    }
    
    p_labels = {
        0.1: 'p=0.1', 0.3: 'p=0.3', 0.5: 'p=0.5',
        0.3536: 'p=1/√8', 0.3162: 'p=1/√10', 0.2887: 'p=1/√12', 
        0.2673: 'p=1/√14', 0.2500: 'p=1/√16'
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes

    # Organiser les données
    organized_data = {}
    for strategy in strategies_to_plot:
        organized_data[strategy] = {}
        for key in keys:
            p = key[1]
            if p not in organized_data[strategy]:
                organized_data[strategy][p] = {'n': [], 'time': [], 'nodes': []}

    # Remplir les données
    for n in ns:
        for strategy in strategies_to_plot:
            for key in keys:
                if key[0] == n:
                    p = key[1]
                    records = all_results[key].get(strategy, [])
                    time_vals = [r['time'] for r in records if r.get('time') and r['time'] > 0 and not r.get('timeout') and 'error' not in r]
                    node_vals = [r['nodes'] for r in records if r.get('nodes') and not r.get('timeout') and 'error' not in r]
                    
                    if time_vals:
                        organized_data[strategy][p]['n'].append(n)
                        organized_data[strategy][p]['time'].append(sum(time_vals) / len(time_vals))
                        organized_data[strategy][p]['nodes'].append(sum(node_vals) / len(node_vals) if node_vals else 0)

    # Tracer
    for strategy in strategies_to_plot:
        color = strategy_colors.get(strategy, 'black')
        strategy_label = strategy_names.get(strategy, strategy)
        
        for p in organized_data[strategy]:
            data = organized_data[strategy][p]
            if not data['n']:
                continue
                
            linestyle = p_linestyles.get(p, '-')
            p_label = p_labels.get(p, f'p={p:.3f}')
            
            # Label complet: stratégie + p
            full_label = f"{strategy_label} ({p_label})"
            
            sorted_indices = sorted(range(len(data['n'])), key=lambda i: data['n'][i])
            sorted_n = [data['n'][i] for i in sorted_indices]
            sorted_time = [data['time'][i] for i in sorted_indices]
            sorted_nodes = [data['nodes'][i] for i in sorted_indices]
            
            log_time = [math.log2(t) if t and t > 0 else 0 for t in sorted_time]

            ax1.plot(sorted_n, log_time, color=color, linestyle=linestyle,
                    marker='o', markersize=4, label=full_label, linewidth=1.5)
            ax2.plot(sorted_n, sorted_nodes, color=color, linestyle=linestyle,
                    marker='o', markersize=4, label=full_label, linewidth=1.5)

    # Configuration
    ax1.set_xlabel('n (nombre de sommets)')
    ax1.set_ylabel('log₂(temps en secondes)')
    ax1.set_title(f'Comparaison complète - Temps {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    ax2.set_xlabel('n (nombre de sommets)')
    ax2.set_ylabel('nœuds générés')
    ax2.set_title(f'Comparaison complète - Nœuds générés {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.show()

def tracer_comparaison_strategies_par_strategie(all_results, strategies_to_plot=None, title_suffix=""):
    """
    Graphique avec sous-graphiques séparés pour chaque stratégie
    """
    keys = list(all_results.keys())
    if not keys:
        print("Aucune donnée à tracer!")
        return
        
    ns = sorted({k[0] for k in keys})
    ps_to_plot = sorted({k[1] for k in keys})
    
    if strategies_to_plot is None:
        example_key = keys[0]
        strategies_to_plot = list(all_results[example_key].keys())

    # Configuration des couleurs et styles
    p_colors = {0.1: 'blue', 0.3: 'red', 0.5: 'green', 
                0.3536: 'orange', 0.3162: 'orange', 0.2887: 'orange', 
                0.2673: 'orange', 0.2500: 'orange'}
    
    p_linestyles = {0.1: '-', 0.3: '--', 0.5: ':', 
                   0.3536: '-.', 0.3162: '-.', 0.2887: '-.', 
                   0.2673: '-.', 0.2500: '-.'}
    
    p_labels = {0.1: 'p=0.1', 0.3: 'p=0.3', 0.5: 'p=0.5',
               0.3536: 'p=1/√8', 0.3162: 'p=1/√10', 0.2887: 'p=1/√12',
               0.2673: 'p=1/√14', 0.2500: 'p=1/√16'}
    
    strategy_names = {
        'simple': 'Branchement simple',
        'couplage_borne': 'Avec couplage et bornes', 
        'glouton': 'Avec glouton seulement',
        'bornes': 'Avec bornes seulement',
        'couplage': 'Avec couplage seulement',
        'glouton_borne': 'Avec glouton et bornes'
    }

    # Créer les sous-graphiques
    n_strategies = len(strategies_to_plot)
    fig, axes = plt.subplots(n_strategies, 2, figsize=(15, 4 * n_strategies))
    
    if n_strategies == 1:
        axes = [axes]

    # Organiser les données
    organized_data = {}
    for strategy in strategies_to_plot:
        organized_data[strategy] = {}
        for p in ps_to_plot:
            organized_data[strategy][p] = {'n': [], 'time': [], 'nodes': []}

    # Remplir les données
    for n in ns:
        for p in ps_to_plot:
            key = (n, p)
            if key not in all_results:
                continue
            for strategy in strategies_to_plot:
                records = all_results[key].get(strategy, [])
                time_vals = [r['time'] for r in records if r.get('time') and r['time'] > 0 and not r.get('timeout') and 'error' not in r]
                node_vals = [r['nodes'] for r in records if r.get('nodes') and not r.get('timeout') and 'error' not in r]
                
                if time_vals:
                    organized_data[strategy][p]['n'].append(n)
                    organized_data[strategy][p]['time'].append(sum(time_vals) / len(time_vals))
                    organized_data[strategy][p]['nodes'].append(sum(node_vals) / len(node_vals) if node_vals else 0)

    # Tracer
    for i, strategy in enumerate(strategies_to_plot):
        ax1, ax2 = axes[i]
        strategy_data = organized_data[strategy]
        
        # Tracer pour chaque p
        for p in ps_to_plot:
            data = strategy_data[p]
            if not data['n']:
                continue
                
            color = p_colors.get(p, 'gray')
            linestyle = p_linestyles.get(p, '-')
            p_label = p_labels.get(p, f'p={p:.3f}')
            
            sorted_indices = sorted(range(len(data['n'])), key=lambda i: data['n'][i])
            sorted_n = [data['n'][i] for i in sorted_indices]
            sorted_time = [data['time'][i] for i in sorted_indices]
            sorted_nodes = [data['nodes'][i] for i in sorted_indices]
            
            log_time = [math.log2(t) if t and t > 0 else 0 for t in sorted_time]

            ax1.plot(sorted_n, log_time, color=color, linestyle=linestyle,
                    marker='o', markersize=4, label=p_label, linewidth=2)
            ax2.plot(sorted_n, sorted_nodes, color=color, linestyle=linestyle,
                    marker='o', markersize=4, label=p_label, linewidth=2)

        # Configuration des sous-graphiques
        strategy_name = strategy_names.get(strategy, strategy)
        ax1.set_xlabel('n (nombre de sommets)')
        ax1.set_ylabel('log₂(temps en secondes)')
        ax1.set_title(f'{strategy_name} - Temps {title_suffix}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        ax2.set_xlabel('n (nombre de sommets)')
        ax2.set_ylabel('nœuds générés')
        ax2.set_title(f'{strategy_name} - Nœuds générés {title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

def tracer_comparaison_strategies_par_p(all_results, strategies_to_plot=None, p_value=0.3, title_suffix=""):
    """
    Graphique pour une valeur de p spécifique
    """
    keys = list(all_results.keys())
    if not keys:
        print("Aucune donnée à tracer!")
        return
        
    # Filtrer pour la valeur de p spécifique
    filtered_keys = [key for key in keys if abs(key[1] - p_value) < 0.01]
    if not filtered_keys:
        print(f"Aucune donnée pour p={p_value}!")
        return
        
    ns = sorted({k[0] for k in filtered_keys})
    
    if strategies_to_plot is None:
        example_key = filtered_keys[0]
        strategies_to_plot = list(all_results[example_key].keys())

    # Couleurs pour les stratégies
    strategy_colors = {
        'simple': 'blue', 'couplage_borne': 'green', 'glouton': 'red',
        'bornes': 'orange', 'couplage': 'purple', 'glouton_borne': 'brown'
    }
    
    strategy_markers = {
        'simple': 'o', 'couplage_borne': 's', 'glouton': '^',
        'bornes': 'D', 'couplage': 'v', 'glouton_borne': '*'
    }
    
    strategy_names = {
        'simple': 'Branchement simple',
        'couplage_borne': 'Avec couplage et bornes', 
        'glouton': 'Avec glouton seulement',
        'bornes': 'Avec bornes seulement',
        'couplage': 'Avec couplage seulement',
        'glouton_borne': 'Avec glouton et bornes'
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # Organiser les données
    organized_data = {s: {'n': [], 'time': [], 'nodes': []} for s in strategies_to_plot}

    for n in ns:
        for strategy in strategies_to_plot:
            key = (n, p_value)
            if key in all_results:
                records = all_results[key].get(strategy, [])
                time_vals = [r['time'] for r in records if r.get('time') and r['time'] > 0 and not r.get('timeout') and 'error' not in r]
                node_vals = [r['nodes'] for r in records if r.get('nodes') and not r.get('timeout') and 'error' not in r]
                
                if time_vals:
                    organized_data[strategy]['n'].append(n)
                    organized_data[strategy]['time'].append(sum(time_vals) / len(time_vals))
                    organized_data[strategy]['nodes'].append(sum(node_vals) / len(node_vals) if node_vals else 0)

    # Tracer
    for strategy in strategies_to_plot:
        data = organized_data[strategy]
        if not data['n']:
            continue
            
        color = strategy_colors.get(strategy, 'black')
        marker = strategy_markers.get(strategy, 'o')
        label = strategy_names.get(strategy, strategy)
        
        sorted_indices = sorted(range(len(data['n'])), key=lambda i: data['n'][i])
        sorted_n = [data['n'][i] for i in sorted_indices]
        sorted_time = [data['time'][i] for i in sorted_indices]
        sorted_nodes = [data['nodes'][i] for i in sorted_indices]
        
        log_time = [math.log2(t) if t and t > 0 else 0 for t in sorted_time]

        ax1.plot(sorted_n, log_time, color=color, marker=marker, 
                linestyle='-', label=label, linewidth=2, markersize=6)
        ax2.plot(sorted_n, sorted_nodes, color=color, marker=marker, 
                linestyle='-', label=label, linewidth=2, markersize=6)

    # Configuration
    ax1.set_xlabel('n (nombre de sommets)')
    ax1.set_ylabel('log₂(temps en secondes)')
    ax1.set_title(f'Comparaison des stratégies (p={p_value}) - Temps {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    ax2.set_xlabel('n (nombre de sommets)')
    ax2.set_ylabel('nœuds générés')
    ax2.set_title(f'Comparaison des stratégies (p={p_value}) - Nœuds générés {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()

def tracer_comparaison_strategies_simple(all_results, strategies_to_plot=None, title_suffix=""):
    """
    Graphique comparatif simple - toutes stratégies sur mêmes axes (moyenne sur tous les p)
    """
    keys = list(all_results.keys())
    if not keys:
        print("Aucune donnée à tracer!")
        return
        
    ns = sorted({k[0] for k in keys})
    
    if strategies_to_plot is None:
        example_key = keys[0]
        strategies_to_plot = list(all_results[example_key].keys())

    # Couleurs pour les stratégies
    strategy_colors = {
        'simple': 'blue', 'couplage_borne': 'green', 'glouton': 'red',
        'bornes': 'orange', 'couplage': 'purple', 'glouton_borne': 'brown'
    }
    
    strategy_markers = {
        'simple': 'o', 'couplage_borne': 's', 'glouton': '^',
        'bornes': 'D', 'couplage': 'v', 'glouton_borne': '*'
    }
    
    strategy_names = {
        'simple': 'Branchement simple',
        'couplage_borne': 'Avec couplage et bornes', 
        'glouton': 'Avec glouton seulement',
        'bornes': 'Avec bornes seulement',
        'couplage': 'Avec couplage seulement',
        'glouton_borne': 'Avec glouton et bornes'
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # Organiser les données (moyenne sur tous les p)
    organized_data = {s: {'n': [], 'time': [], 'nodes': []} for s in strategies_to_plot}

    for n in ns:
        for strategy in strategies_to_plot:
            time_vals = []
            node_vals = []
            
            # Prendre la moyenne sur tous les p pour ce n
            for key in keys:
                if key[0] == n:
                    records = all_results[key].get(strategy, [])
                    time_vals.extend([r['time'] for r in records if r.get('time') and r['time'] > 0 and not r.get('timeout') and 'error' not in r])
                    node_vals.extend([r['nodes'] for r in records if r.get('nodes') and not r.get('timeout') and 'error' not in r])
            
            if time_vals:
                organized_data[strategy]['n'].append(n)
                organized_data[strategy]['time'].append(sum(time_vals) / len(time_vals))
                organized_data[strategy]['nodes'].append(sum(node_vals) / len(node_vals) if node_vals else 0)

    # Tracer
    for strategy in strategies_to_plot:
        data = organized_data[strategy]
        if not data['n']:
            continue
            
        color = strategy_colors.get(strategy, 'black')
        marker = strategy_markers.get(strategy, 'o')
        label = strategy_names.get(strategy, strategy)
        
        sorted_indices = sorted(range(len(data['n'])), key=lambda i: data['n'][i])
        sorted_n = [data['n'][i] for i in sorted_indices]
        sorted_time = [data['time'][i] for i in sorted_indices]
        sorted_nodes = [data['nodes'][i] for i in sorted_indices]
        
        log_time = [math.log2(t) if t and t > 0 else 0 for t in sorted_time]

        ax1.plot(sorted_n, log_time, color=color, marker=marker, 
                linestyle='-', label=label, linewidth=2, markersize=6)
        ax2.plot(sorted_n, sorted_nodes, color=color, marker=marker, 
                linestyle='-', label=label, linewidth=2, markersize=6)

    # Configuration
    ax1.set_xlabel('n (nombre de sommets)')
    ax1.set_ylabel('log₂(temps en secondes)')
    ax1.set_title(f'Comparaison simplifiée - Temps {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    ax2.set_xlabel('n (nombre de sommets)')
    ax2.set_ylabel('nœuds générés')
    ax2.set_title(f'Comparaison simplifiée - Nœuds générés {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.show()

def tester_branchement_ameliores(n_values, p_values_labels, num_instances=3, max_time_par_instance=30):
    """
    Teste spécifiquement les trois versions améliorées du branchement
    """
    strategies_to_run = ['v1', 'v2', 'v3']
    
    strategy_methods = {
        'v1': 'branchement_ameliore_v1',
        'v2': 'branchement_ameliore_v2', 
        'v3': 'branchement_ameliore_v3'
    }
    
    strategy_names = {
        'v1': 'Branchement amélioré v1',
        'v2': 'Branchement amélioré v2 (degré max)',
        'v3': 'Branchement amélioré v3 (degré 1)'
    }

    all_results = {}

    for n in n_values:
        for p_label in p_values_labels:
            if p_label == '1/sqrt':
                p = 1.0 / math.sqrt(n)
            else:
                p = float(p_label)
            
            key = (n, p)
            all_results[key] = {s: [] for s in strategies_to_run}

            for inst in range(num_instances):
                G = generate_random_graph(n, p)
                print(f"n={n}, p={p:.4f}, inst={inst+1}/{num_instances}")

                for strategy in strategies_to_run:
                    method_name = strategy_methods[strategy]
                    method = getattr(G, method_name)
                    
                    t0 = time.time()
                    try:
                        coverage, nodes = method()
                        t1 = time.time()
                        
                        valid = G.est_couverture_valide(coverage)
                        timeout = (t1 - t0) > max_time_par_instance
                        
                        all_results[key][strategy].append({
                            'time': t1 - t0,
                            'nodes': nodes,
                            'size': len(coverage),
                            'valid': valid,
                            'timeout': timeout
                        })
                        
                    except Exception as e:
                        all_results[key][strategy].append({
                            'time': None, 'nodes': None, 'size': None, 
                            'valid': False, 'error': str(e), 'timeout': False
                        })

    return all_results, strategy_names

def tracer_comparaison_ameliores(all_results, strategy_names, title_suffix=""):
    """
    Trace la comparaison des trois versions améliorées du branchement
    """
    keys = list(all_results.keys())
    ns = sorted({k[0] for k in keys})
    
    strategies_to_plot = list(strategy_names.keys())
    
    # Couleurs pour les différentes stratégies
    strategy_colors = {
        'v1': 'blue',
        'v2': 'green', 
        'v3': 'red'
    }
    
    strategy_markers = {
        'v1': 'o',
        'v2': 's',
        'v3': '^'
    }

    # Créer les figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_time, ax_nodes, ax_size, ax_ratio = axes.flatten()

    # Organiser les données par stratégie
    organized_data = {s: {'n': [], 'time': [], 'nodes': [], 'size': []} for s in strategies_to_plot}

    for n in ns:
        for strategy in strategies_to_plot:
            time_vals = []
            node_vals = []
            size_vals = []
            
            for p in [0.1, 0.3, 0.5, '1/sqrt']:
                if p == '1/sqrt':
                    actual_p = 1.0 / math.sqrt(n)
                else:
                    actual_p = p
                    
                key = (n, actual_p)
                if key in all_results:
                    records = all_results[key].get(strategy, [])
                    for record in records:
                        if record.get('time') is not None and record['time'] > 0:
                            time_vals.append(record['time'])
                        if record.get('nodes') is not None:
                            node_vals.append(record['nodes'])
                        if record.get('size') is not None:
                            size_vals.append(record['size'])
            
            if time_vals:
                organized_data[strategy]['n'].append(n)
                organized_data[strategy]['time'].append(sum(time_vals) / len(time_vals))
                organized_data[strategy]['nodes'].append(sum(node_vals) / len(node_vals) if node_vals else 0)
                organized_data[strategy]['size'].append(sum(size_vals) / len(size_vals) if size_vals else 0)

    # Tracer les données
    for strategy in strategies_to_plot:
        data = organized_data[strategy]
        if not data['n']:
            continue
            
        color = strategy_colors[strategy]
        marker = strategy_markers[strategy]
        label = strategy_names[strategy]
        
        # Trier les données par n
        sorted_data = sorted(zip(data['n'], data['time'], data['nodes'], data['size']))
        sorted_n = [item[0] for item in sorted_data]
        sorted_time = [item[1] for item in sorted_data]
        sorted_nodes = [item[2] for item in sorted_data]
        sorted_size = [item[3] for item in sorted_data]
        
        # Temps d'exécution
        ax_time.plot(sorted_n, sorted_time, color=color, marker=marker, 
                    linestyle='-', label=label, linewidth=2, markersize=6)
        
        # Nœuds générés (échelle log)
        log_nodes = [math.log(nodes) if nodes and nodes > 0 else 0 for nodes in sorted_nodes]
        ax_nodes.plot(sorted_n, log_nodes, color=color, marker=marker,
                     linestyle='-', label=label, linewidth=2, markersize=6)
        
        # Taille de la couverture
        ax_size.plot(sorted_n, sorted_size, color=color, marker=marker,
                    linestyle='-', label=label, linewidth=2, markersize=6)
        
        # Ratio par rapport à v1
        if strategy != 'v1' and 'v1' in strategies_to_plot:
            ratio_times = []
            for i, t_v1 in enumerate(organized_data['v1']['time']):
                if i < len(sorted_time):
                    t_current = sorted_time[i]
                    if t_v1 > 0 and t_current > 0:
                        ratio_times.append(t_current / t_v1)
                    else:
                        ratio_times.append(float('nan'))
            ax_ratio.plot(sorted_n, ratio_times, color=color, marker=marker,
                         linestyle='-', label=label, linewidth=2, markersize=6)

    # Configurer les axes
    ax_time.set_xlabel('n (nombre de sommets)')
    ax_time.set_ylabel('Temps moyen (s)')
    ax_time.set_title(f'Temps d\'exécution {title_suffix}')
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    ax_nodes.set_xlabel('n (nombre de sommets)')
    ax_nodes.set_ylabel('log(Nœuds générés)')
    ax_nodes.set_title(f'Nœuds générés (échelle log) {title_suffix}')
    ax_nodes.grid(True, alpha=0.3)
    ax_nodes.legend()

    ax_size.set_xlabel('n (nombre de sommets)')
    ax_size.set_ylabel('Taille moyenne de la couverture')
    ax_size.set_title(f'Qualité des solutions {title_suffix}')
    ax_size.grid(True, alpha=0.3)
    ax_size.legend()

    ax_ratio.set_xlabel('n (nombre de sommets)')
    ax_ratio.set_ylabel('Ratio de temps (vs v1)')
    ax_ratio.set_title(f'Ratio de temps vs branchement amélioré v1 {title_suffix}')
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend()

    plt.tight_layout()
    plt.show()

    # Afficher un résumé statistique
    print("\n=== RÉSUMÉ STATISTIQUE DES BRANCHEMENTS AMÉLIORÉS ===")
    for strategy in strategies_to_plot:
        print(f"\n{strategy_names[strategy]}:")
        total_time = 0
        total_nodes = 0
        total_size = 0
        count = 0
        
        for key in keys:
            records = all_results[key].get(strategy, [])
            for record in records:
                if record.get('time') is not None:
                    total_time += record['time']
                if record.get('nodes') is not None:
                    total_nodes += record['nodes']
                if record.get('size') is not None:
                    total_size += record['size']
                count += 1
        
        if count > 0:
            avg_time = total_time / count
            avg_nodes = total_nodes / count
            avg_size = total_size / count
            print(f"  Temps moyen: {avg_time:.4f}s")
            print(f"  Nœuds moyens: {avg_nodes:.0f}")
            print(f"  Taille moyenne de couverture: {avg_size:.2f}")

def evaluer_qualite_approximation(n_values, p_values, num_instances=10, max_n_optimal=100):
    """
    Évalue expérimentalement le rapport d'approximation des algorithmes de couplage et glouton.
    Utilise branchement_ameliore_v3 pour obtenir la solution optimale.
    
    Args:
        n_values: Liste des tailles de graphes à tester
        p_values: Liste des probabilités d'arêtes à tester
        num_instances: Nombre d'instances par configuration
        max_n_optimal: Taille maximale pour laquelle calculer la solution optimale
        
    Returns:
        dict: Résultats détaillés pour chaque configuration
    """
    results = {}
    
    for n in n_values:
        for p in p_values:
            key = (n, p)
            results[key] = {
                'glouton_ratios': [],
                'couplage_ratios': [],
                'glouton_worst': 0,
                'couplage_worst': 0,
                'optimal_sizes': [],
                'glouton_sizes': [],
                'couplage_sizes': []
            }
            
            print(f"Test n={n}, p={p}")
            
            for inst in range(num_instances):
                # Générer un graphe aléatoire
                G = generate_random_graph(n, p)
                
                # Calculer les couvertures avec les algorithmes approximatifs
                couverture_glouton = G.algo_glouton()
                couverture_couplage = G.algo_couplage()
                
                # Enregistrer les tailles des couvertures approximatives
                results[key]['glouton_sizes'].append(len(couverture_glouton))
                results[key]['couplage_sizes'].append(len(couverture_couplage))
                
                # Calculer la couverture optimale avec branchement_ameliore_v3 pour les petits graphes
                if n <= max_n_optimal:
                    try:
                        # Utiliser branchement_ameliore_v3 pour la solution optimale
                        couverture_optimale, _ = G.branchement_ameliore_v3()
                        taille_optimale = len(couverture_optimale)
                        
                        # Calculer les ratios d'approximation
                        if taille_optimale > 0:
                            ratio_glouton = len(couverture_glouton) / taille_optimale
                            ratio_couplage = len(couverture_couplage) / taille_optimale
                        else:
                            ratio_glouton = 1.0
                            ratio_couplage = 1.0
                        
                        # Mettre à jour les résultats
                        results[key]['glouton_ratios'].append(ratio_glouton)
                        results[key]['couplage_ratios'].append(ratio_couplage)
                        results[key]['optimal_sizes'].append(taille_optimale)
                        
                        # Mettre à jour les pires ratios
                        results[key]['glouton_worst'] = max(results[key]['glouton_worst'], ratio_glouton)
                        results[key]['couplage_worst'] = max(results[key]['couplage_worst'], ratio_couplage)
                        
                        print(f"  Instance {inst+1}: optimal={taille_optimale}, glouton={len(couverture_glouton)}, couplage={len(couverture_couplage)}")
                    
                    except Exception as e:
                        print(f"  Erreur lors du calcul de la couverture optimale pour n={n}: {e}")
                        # En cas d'erreur, on considère que les ratios sont 1.0
                        results[key]['glouton_ratios'].append(1.0)
                        results[key]['couplage_ratios'].append(1.0)
                        results[key]['optimal_sizes'].append(0)
                else:
                    # Pour les grands graphes, on ne calcule pas le rapport exact
                    results[key]['glouton_ratios'].append(None)
                    results[key]['couplage_ratios'].append(None)
                    results[key]['optimal_sizes'].append(None)
                    print(f"  Instance {inst+1}: glouton={len(couverture_glouton)}, couplage={len(couverture_couplage)} (optimal non calculé pour n>{max_n_optimal})")
    
    return results

def tracer_rapports_approximation(results, title_suffix="", max_n_optimal=100):
    """
    Trace les rapports d'approximation en fonction de n
    Utilise les données calculées avec branchement_ameliore_v3 comme référence optimale.
    """
    # Organiser les données par n et p
    n_values = sorted({key[0] for key in results.keys()})
    p_values = sorted({key[1] for key in results.keys()})
    
    # Créer les figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Couleurs pour différentes valeurs de p
    p_colors = {
        0.1: 'blue',
        0.3: 'green',
        0.5: 'red',
        0.7: 'orange'
    }
    
    p_markers = {
        0.1: 'o',
        0.3: 's',
        0.5: '^',
        0.7: 'D'
    }
    
    # Préparer les données pour les graphiques
    data_by_p = {}
    for p in p_values:
        data_by_p[p] = {
            'n': [],
            'glouton_mean': [],
            'couplage_mean': [],
            'glouton_worst': [],
            'couplage_worst': [],
            'glouton_sizes': [],
            'couplage_sizes': [],
            'optimal_sizes': []
        }
    
    # Organiser les données
    for n in n_values:
        for p in p_values:
            key = (n, p)
            if key in results:
                data = results[key]
                
                # Données pour les rapports (seulement pour n <= max_n_optimal)
                if n <= max_n_optimal and data['glouton_ratios'] and any(r is not None for r in data['glouton_ratios']):
                    valid_glouton_ratios = [r for r in data['glouton_ratios'] if r is not None]
                    valid_couplage_ratios = [r for r in data['couplage_ratios'] if r is not None]
                    
                    if valid_glouton_ratios and valid_couplage_ratios:
                        data_by_p[p]['n'].append(n)
                        data_by_p[p]['glouton_mean'].append(sum(valid_glouton_ratios) / len(valid_glouton_ratios))
                        data_by_p[p]['couplage_mean'].append(sum(valid_couplage_ratios) / len(valid_couplage_ratios))
                        data_by_p[p]['glouton_worst'].append(data['glouton_worst'])
                        data_by_p[p]['couplage_worst'].append(data['couplage_worst'])
                
                # Données pour les tailles (tous les n)
                if data['glouton_sizes'] and data['couplage_sizes']:
                    # Stocker les données de taille pour analyse
                    data_by_p[p]['glouton_sizes'].extend(data['glouton_sizes'])
                    data_by_p[p]['couplage_sizes'].extend(data['couplage_sizes'])
                    if data['optimal_sizes'] and any(s is not None for s in data['optimal_sizes']):
                        valid_optimal = [s for s in data['optimal_sizes'] if s is not None]
                        if valid_optimal:
                            data_by_p[p]['optimal_sizes'].extend(valid_optimal)
    
    # Tracer les rapports moyens d'approximation (seulement pour n <= max_n_optimal)
    for p in p_values:
        if data_by_p[p]['n']:  # Vérifier qu'il y a des données à tracer
            color = p_colors.get(p, 'black')
            marker = p_markers.get(p, 'o')
            label = f'p={p}'
            
            # Rapport moyen - Glouton
            line1, = ax1.plot(data_by_p[p]['n'], data_by_p[p]['glouton_mean'], 
                    color=color, marker=marker, linestyle='-', 
                    label=f'Glouton ({label})', linewidth=2, markersize=6)
            
            # Rapport moyen - Couplage
            line2, = ax2.plot(data_by_p[p]['n'], data_by_p[p]['couplage_mean'], 
                    color=color, marker=marker, linestyle='-', 
                    label=f'Couplage ({label})', linewidth=2, markersize=6)
            
            # Pire rapport - Glouton
            line3, = ax3.plot(data_by_p[p]['n'], data_by_p[p]['glouton_worst'], 
                    color=color, marker=marker, linestyle='-', 
                    label=f'Glouton ({label})', linewidth=2, markersize=6)
            
            # Pire rapport - Couplage
            line4, = ax4.plot(data_by_p[p]['n'], data_by_p[p]['couplage_worst'], 
                    color=color, marker=marker, linestyle='-', 
                    label=f'Couplage ({label})', linewidth=2, markersize=6)
    
    # Configurer les graphiques seulement s'il y a des données
    if any(data_by_p[p]['n'] for p in p_values):
        ax1.set_xlabel('n (nombre de sommets)')
        ax1.set_ylabel('Rapport d\'approximation moyen')
        ax1.set_title(f'Rapport d\'approximation moyen - Algorithme glouton {title_suffix}')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Borne théorique (2)')
        if ax1.get_legend_handles_labels()[0]:
            ax1.legend()
        
        ax2.set_xlabel('n (nombre de sommets)')
        ax2.set_ylabel('Rapport d\'approximation moyen')
        ax2.set_title(f'Rapport d\'approximation moyen - Algorithme de couplage {title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Borne théorique (2)')
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        
        ax3.set_xlabel('n (nombre de sommets)')
        ax3.set_ylabel('Pire rapport d\'approximation')
        ax3.set_title(f'Pire rapport d\'approximation - Algorithme glouton {title_suffix}')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Borne théorique (2)')
        if ax3.get_legend_handles_labels()[0]:
            ax3.legend()
        
        ax4.set_xlabel('n (nombre de sommets)')
        ax4.set_ylabel('Pire rapport d\'approximation')
        ax4.set_title(f'Pire rapport d\'approximation - Algorithme de couplage {title_suffix}')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Borne théorique (2)')
        if ax4.get_legend_handles_labels()[0]:
            ax4.legend()
    else:
        print("Aucune donnée à tracer pour les rapports d'approximation (n trop grand pour le calcul optimal).")
    
    plt.tight_layout()
    plt.show()
    
    # Afficher un résumé statistique
    print("\n" + "="*80)
    print("RÉSUMÉ STATISTIQUE DES RAPPORTS D'APPROXIMATION")
    print("(Basé sur branchement_ameliore_v3 pour la solution optimale)")
    print("="*80)
    
    # Calculer les pires rapports globaux (uniquement pour les petits graphes)
    glouton_worst_global = 0
    couplage_worst_global = 0
    glouton_worst_config = None
    couplage_worst_config = None
    
    for key, data in results.items():
        n = key[0]
        if n <= max_n_optimal:  # Seulement pour les petits graphes où nous avons calculé les ratios
            if data['glouton_worst'] > glouton_worst_global:
                glouton_worst_global = data['glouton_worst']
                glouton_worst_config = key
            
            if data['couplage_worst'] > couplage_worst_global:
                couplage_worst_global = data['couplage_worst']
                couplage_worst_config = key
    
    if glouton_worst_config:
        print(f"\nPire rapport d'approximation - Algorithme glouton: {glouton_worst_global:.4f}")
        print(f"  Configuration: n={glouton_worst_config[0]}, p={glouton_worst_config[1]}")
    else:
        print(f"\nPire rapport d'approximation - Algorithme glouton: Non calculé (n > {max_n_optimal})")
    
    if couplage_worst_config:
        print(f"\nPire rapport d'approximation - Algorithme de couplage: {couplage_worst_global:.4f}")
        print(f"  Configuration: n={couplage_worst_config[0]}, p={couplage_worst_config[1]}")
    else:
        print(f"\nPire rapport d'approximation - Algorithme de couplage: Non calculé (n > {max_n_optimal})")
    
    # Afficher les rapports moyens par algorithme (uniquement pour n <= max_n_optimal)
    print(f"\nRapports d'approximation moyens par configuration (n ≤ {max_n_optimal}):")
    print("-" * 60)
    print("Configuration\t\tGlouton (moyen)\tCouplage (moyen)")
    print("-" * 60)
    
    for n in n_values:
        if n <= max_n_optimal:  # Seulement pour les petits graphes
            for p in p_values:
                key = (n, p)
                if key in results and results[key]['glouton_ratios']:
                    valid_glouton = [r for r in results[key]['glouton_ratios'] if r is not None]
                    valid_couplage = [r for r in results[key]['couplage_ratios'] if r is not None]
                    
                    if valid_glouton and valid_couplage:
                        glouton_mean = sum(valid_glouton) / len(valid_glouton)
                        couplage_mean = sum(valid_couplage) / len(valid_couplage)
                        print(f"n={n}, p={p}\t\t{glouton_mean:.4f}\t\t{couplage_mean:.4f}")

def evaluer_heuristiques(n_values, p_values, num_instances=10):
    """
    Évalue expérimentalement différentes heuristiques
    
    Args:
        n_values: Liste des tailles de graphes
        p_values: Liste des probabilités d'arêtes
        num_instances: Nombre d'instances par configuration
        
    Returns:
        dict: Résultats pour chaque heuristique
    """
    heuristiques = {
        'glouton': lambda G: G.algo_glouton(),
        'couplage': lambda G: G.algo_couplage(),
        'aleatoire': lambda G: G.heuristique_aleatoire(50),
        'degre_pondere': lambda G: G.heuristique_degre_ponderee(30),
        'recherche_locale': lambda G: G.heuristique_recherche_locale(max_iter=300),
        'hybride': lambda G: G.heuristique_hybride()
    }
    
    noms_heuristiques = {
        'glouton': 'Algorithme glouton',
        'couplage': 'Algorithme de couplage',
        'aleatoire': 'Heuristique aléatoire',
        'degre_pondere': 'Heuristique degré pondéré',
        'recherche_locale': 'Recherche locale',
        'hybride': 'Heuristique hybride'
    }
    
    results = {}
    
    for n in n_values:
        for p in p_values:
            key = (n, p)
            results[key] = {heur: {'sizes': [], 'temps': []} for heur in heuristiques.keys()}
            
            print(f"Test n={n}, p={p}")
            
            for inst in range(num_instances):
                G = generate_random_graph(n, p)
                
                for heur_name, heur_func in heuristiques.items():
                    start_time = time.time()
                    couverture = heur_func(G)
                    temps = time.time() - start_time
                    
                    if G.est_couverture_valide(couverture):
                        results[key][heur_name]['sizes'].append(len(couverture))
                        results[key][heur_name]['temps'].append(temps)
                    else:
                        # Si la solution n'est pas valide, utiliser une valeur pénalisante
                        results[key][heur_name]['sizes'].append(n)  # pire cas: tous les sommets
                        results[key][heur_name]['temps'].append(temps)
    
    return results, noms_heuristiques

def tracer_comparaison_heuristiques(results, noms_heuristiques, title_suffix=""):
    """
    Trace la comparaison des différentes heuristiques
    """
    n_values = sorted({key[0] for key in results.keys()})
    p_values = sorted({key[1] for key in results.keys()})
    heuristiques = list(noms_heuristiques.keys())
    
    # Couleurs pour les heuristiques
    couleurs = {
        'glouton': 'red',
        'couplage': 'blue',
        'aleatoire': 'green',
        'degre_pondere': 'orange',
        'recherche_locale': 'purple',
        'hybride': 'brown'
    }
    
    marqueurs = {
        'glouton': 'o',
        'couplage': 's',
        'aleatoire': '^',
        'degre_pondere': 'D',
        'recherche_locale': 'v',
        'hybride': '*'
    }
    
    # Créer les figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Organiser les données
    donnees_par_p = {}
    for p in p_values:
        donnees_par_p[p] = {heur: {'n': [], 'taille_moy': [], 'temps_moy': []} for heur in heuristiques}
    
    for n in n_values:
        for p in p_values:
            key = (n, p)
            if key in results:
                for heur in heuristiques:
                    if results[key][heur]['sizes']:
                        donnees_par_p[p][heur]['n'].append(n)
                        donnees_par_p[p][heur]['taille_moy'].append(
                            sum(results[key][heur]['sizes']) / len(results[key][heur]['sizes'])
                        )
                        donnees_par_p[p][heur]['temps_moy'].append(
                            sum(results[key][heur]['temps']) / len(results[key][heur]['temps'])
                        )
    
    # Tracer pour p=0.3 (valeur typique)
    p_a_tracer = 0.3
    if p_a_tracer in p_values:
        for heur in heuristiques:
            if donnees_par_p[p_a_tracer][heur]['n']:
                couleur = couleurs.get(heur, 'black')
                marqueur = marqueurs.get(heur, 'o')
                nom = noms_heuristiques[heur]
                
                # Taille moyenne de la couverture
                ax1.plot(donnees_par_p[p_a_tracer][heur]['n'], 
                        donnees_par_p[p_a_tracer][heur]['taille_moy'],
                        color=couleur, marker=marqueur, linestyle='-',
                        label=nom, linewidth=2, markersize=6)
                
                # Temps d'exécution
                ax2.plot(donnees_par_p[p_a_tracer][heur]['n'], 
                        donnees_par_p[p_a_tracer][heur]['temps_moy'],
                        color=couleur, marker=marqueur, linestyle='-',
                        label=nom, linewidth=2, markersize=6)
    
    # Tracer le ratio par rapport à la meilleure heuristique
    for n in n_values:
        for p in [0.3]:  # On se concentre sur p=0.3
            key = (n, p)
            if key in results:
                # Trouver la meilleure taille moyenne
                meilleures_tailles = {}
                for heur in heuristiques:
                    if results[key][heur]['sizes']:
                        meilleures_tailles[heur] = sum(results[key][heur]['sizes']) / len(results[key][heur]['sizes'])
                
                if meilleures_tailles:
                    meilleure_taille = min(meilleures_tailles.values())
                    
                    for heur in heuristiques:
                        if heur in meilleures_tailles:
                            ratio = meilleures_tailles[heur] / meilleure_taille
                            couleur = couleurs.get(heur, 'black')
                            marqueur = marqueurs.get(heur, 'o')
                            nom = noms_heuristiques[heur]
                            
                            # Tracer le point
                            ax3.scatter(n, ratio, color=couleur, marker=marqueur, s=60, label=nom if n == n_values[0] else "")
    
    # Diagramme en barres pour les performances relatives
    tailles_moyennes_globales = {}
    for heur in heuristiques:
        tailles = []
        for key in results:
            if results[key][heur]['sizes']:
                tailles.extend(results[key][heur]['sizes'])
        if tailles:
            tailles_moyennes_globales[heur] = sum(tailles) / len(tailles)
    
    if tailles_moyennes_globales:
        meilleure_taille_globale = min(tailles_moyennes_globales.values())
        ratios_globaux = {heur: taille / meilleure_taille_globale 
                         for heur, taille in tailles_moyennes_globales.items()}
        
        noms_affiches = [noms_heuristiques[heur] for heur in heuristiques if heur in ratios_globaux]
        valeurs = [ratios_globaux[heur] for heur in heuristiques if heur in ratios_globaux]
        couleurs_affichees = [couleurs[heur] for heur in heuristiques if heur in ratios_globaux]
        
        bars = ax4.bar(noms_affiches, valeurs, color=couleurs_affichees, alpha=0.7)
        ax4.set_ylabel('Ratio par rapport à la meilleure')
        ax4.set_title('Performance relative globale')
        
        # Ajouter les valeurs sur les barres
        for bar, valeur in zip(bars, valeurs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{valeur:.3f}', ha='center', va='bottom', rotation=0)
    
    # Configurer les graphiques
    ax1.set_xlabel('n (nombre de sommets)')
    ax1.set_ylabel('Taille moyenne de la couverture')
    ax1.set_title(f'Taille des couvertures (p={p_a_tracer}) {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('n (nombre de sommets)')
    ax2.set_ylabel('Temps d\'exécution moyen (s)')
    ax2.set_title(f'Temps d\'exécution (p={p_a_tracer}) {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')  # Échelle logarithmique pour mieux voir les différences
    
    ax3.set_xlabel('n (nombre de sommets)')
    ax3.set_ylabel('Ratio par rapport à la meilleure')
    ax3.set_title(f'Ratio de performance (p={p_a_tracer}) {title_suffix}')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher un résumé statistique
    print("\n" + "="*80)
    print("RÉSUMÉ STATISTIQUE DES HEURISTIQUES")
    print("="*80)
    
    # Calculer les statistiques globales
    stats_globales = {}
    for heur in heuristiques:
        toutes_tailles = []
        tous_temps = []
        
        for key in results:
            if results[key][heur]['sizes']:
                toutes_tailles.extend(results[key][heur]['sizes'])
                tous_temps.extend(results[key][heur]['temps'])
        
        if toutes_tailles:
            stats_globales[heur] = {
                'taille_moyenne': sum(toutes_tailles) / len(toutes_tailles),
                'temps_moyen': sum(tous_temps) / len(tous_temps),
                'meilleure_taille': min(toutes_tailles),
                'pire_taille': max(toutes_tailles)
            }
    
    # Afficher le tableau des statistiques
    print(f"\n{'Heuristique':<25} {'Taille moy':<12} {'Temps moy (s)':<14} {'Meilleure':<10} {'Pire':<10}")
    print("-" * 75)
    
    for heur in heuristiques:
        if heur in stats_globales:
            stats = stats_globales[heur]
            print(f"{noms_heuristiques[heur]:<25} {stats['taille_moyenne']:<12.2f} {stats['temps_moyen']:<14.4f} {stats['meilleure_taille']:<10} {stats['pire_taille']:<10}")
 
def tester_algorithme_universel_ameliore(current_graph=None):
    """
    Fonction universelle améliorée pour tester n'importe quel algorithme.
    Utilise le graphe courant si disponible, sinon propose d'en charger un.
    
    Args:
        current_graph: Graphe courant mémorisé (optionnel)
        
    Returns:
        Graph: Le graphe courant (pour mémorisation)
    """
    graphe = current_graph
    
    # Si pas de graphe courant, demander à en charger un
    if graphe is None:
        filename = input("Entrez le chemin du fichier contenant le graphe: ").strip()
        try:
            graphe = Graph(read_graph(filename))
            print(f"✓ Graphe chargé depuis {filename}")
            print(f"  • Sommets: {len(graphe.sommets())}")
            print(f"  • Arêtes: {len(graphe.aretes())}")
            print(f"  • Degré maximal: {graphe.max_degree_vertex(return_all=False)[1]}")
        except FileNotFoundError:
            print(f"✗ Fichier non trouvé: {filename}")
            return None
        except Exception as e:
            print(f"✗ Erreur lors du chargement du graphe: {e}")
            return None
    else:
        print("✓ Utilisation du graphe courant")
        print(f"  • Sommets: {len(graphe.sommets())}")
        print(f"  • Arêtes: {len(graphe.aretes())}")
        print(f"  • Degré maximal: {graphe.max_degree_vertex(return_all=False)[1]}")
        
        # Option pour recharger un autre fichier
        changer = input("\nVoulez-vous charger un autre fichier? (o/n): ").strip().lower()
        if changer == 'o':
            filename = input("Entrez le nouveau chemin du fichier: ").strip()
            try:
                graphe = Graph(read_graph(filename))
                print(f"✓ Nouveau graphe chargé depuis {filename}")
                print(f"  • Sommets: {len(graphe.sommets())}")
                print(f"  • Arêtes: {len(graphe.aretes())}")
            except Exception as e:
                print(f"✗ Erreur lors du chargement: {e}")
                return current_graph  # Garder l'ancien graphe en cas d'erreur
    
    # Menu principal des algorithmes
    while True:
        print("\n" + "="*60)
        print("MENU DES ALGORITHMES - GRAPHE COURANT")
        print("="*60)
        print(f"Graphe: {len(graphe.sommets())} sommets, {len(graphe.aretes())} arêtes")
        
        print("\n🔹 ALGORITHMES APPROXIMATIFS:")
        print("   1. Algorithme de couplage")
        print("   2. Algorithme glouton")
        
        print("\n🔹 ALGORITHMES EXACTS (BRANCHEMENT):")
        print("   3. Branchement simple")
        print("   4. Branchement avec couplage et bornes")
        print("   5. Branchement avec glouton seulement")
        print("   6. Branchement avec bornes seulement")
        print("   7. Branchement avec couplage seulement")
        print("   8. Branchement avec glouton et bornes")
        
        print("\n🔹 BRANCHEMENTS AMÉLIORÉS:")
        print("   9. Branchement amélioré v1")
        print("   10. Branchement amélioré v2 (degré max)")
        print("   11. Branchement amélioré v3 (degré 1)")
        
        print("\n🔹 HEURISTIQUES AVANCÉES:")
        print("   12. Heuristique aléatoire")
        print("   13. Heuristique degré pondéré")
        print("   14. Heuristique recherche locale")
        print("   15. Heuristique hybride")
        
        print("\n🔹 ALGORITHME EXACT (PETITS GRAPHES):")
        print("   16. Force brute")
        
        print("\n🔹 COMPARAISONS MULTIPLES:")
        print("   17. Tester tous les algorithmes approximatifs")
        print("   18. Tester tous les algorithmes de branchement")
        print("   19. Comparaison complète (tous les algorithmes)")
        
        print("\n🔹 OPTIONS:")
        print("   20. Changer de graphe")
        print("   21. Retour au menu principal")
        print("-"*60)
        
        choix = input("Choisissez une option (1-21): ").strip()
        
        if choix == "21":
            return graphe  # Retourner le graphe courant pour mémorisation
        
        elif choix == "20":
            # Changer de graphe
            filename = input("Entrez le nouveau chemin du fichier: ").strip()
            try:
                graphe = Graph(read_graph(filename))
                print(f"✓ Nouveau graphe chargé depuis {filename}")
                print(f"  • Sommets: {len(graphe.sommets())}")
                print(f"  • Arêtes: {len(graphe.aretes())}")
                continue
            except Exception as e:
                print(f"✗ Erreur lors du chargement: {e}")
                continue
        
        # Mapping des algorithmes individuels
        algorithmes_individuels = {
            '1': ("Algorithme de couplage", graphe.algo_couplage, False),
            '2': ("Algorithme glouton", graphe.algo_glouton, False),
            '3': ("Branchement simple", graphe.branchement_simple, True),
            '4': ("Branchement avec couplage et bornes", graphe.branchement_couplage_avec_borne, True),
            '5': ("Branchement avec glouton seulement", graphe.branchement_avec_glouton_seulement, True),
            '6': ("Branchement avec bornes seulement", graphe.branchement_avec_bornes_seulement, True),
            '7': ("Branchement avec couplage seulement", graphe.branchement_avec_couplage_seulement, True),
            '8': ("Branchement avec glouton et bornes", graphe.branchement_glouton_avec_borne, True),
            '9': ("Branchement amélioré v1", graphe.branchement_ameliore_v1, True),
            '10': ("Branchement amélioré v2", graphe.branchement_ameliore_v2, True),
            '11': ("Branchement amélioré v3", graphe.branchement_ameliore_v3, True),
            '12': ("Heuristique aléatoire", graphe.heuristique_aleatoire, False),
            '13': ("Heuristique degré pondéré", graphe.heuristique_degre_ponderee, False),
            '14': ("Heuristique recherche locale", graphe.heuristique_recherche_locale, False),
            '15': ("Heuristique hybride", graphe.heuristique_hybride, False),
            '16': ("Force brute", lambda: bruteforce_vertex_cover(graphe.adj), False)
        }
        
        # Exécution d'un algorithme individuel
        if choix in algorithmes_individuels:
            nom_algo, fonction_algo, retourne_noeuds = algorithmes_individuels[choix]
            executer_algorithme_individuel(graphe, nom_algo, fonction_algo, retourne_noeuds)
        
        # Comparaisons multiples
        elif choix == "17":
            tester_tous_algorithmes_approximatifs(graphe)
        
        elif choix == "18":
            tester_tous_algorithmes_branchement(graphe)
        
        elif choix == "19":
            tester_tous_algorithmes(graphe)
        
        else:
            print("✗ Option invalide!")
        
        input("\nAppuyez sur Entrée pour continuer...")

def executer_algorithme_individuel(graphe, nom_algo, fonction_algo, retourne_noeuds):
    """
    Exécute un algorithme individuel et affiche les résultats
    """
    # Avertissements pour les algorithmes lents
    n_sommets = len(graphe.sommets())
    
    # Vérifications de sécurité pour les algorithmes lents
    if "branchement" in nom_algo.lower() and n_sommets > 20:
        print(f"⚠️  ATTENTION: Le graphe a {n_sommets} sommets.")
        print("   Les algorithmes de branchement peuvent être très lents.")
        reponse = input("   Voulez-vous continuer? (o/n): ").strip().lower()
        if reponse != 'o':
            return
    
    if "force brute" in nom_algo.lower() and n_sommets > 15:
        print(f"⚠️  ATTENTION: Le graphe a {n_sommets} sommets.")
        print("   La force brute sera extrêmement lente (complexité exponentielle).")
        reponse = input("   Voulez-vous vraiment continuer? (o/n): ").strip().lower()
        if reponse != 'o':
            return
    
    # Exécution de l'algorithme
    print(f"\n🎯 Exécution de: {nom_algo}")
    print("-" * 40)
    
    debut = time.time()
    
    try:
        resultat = fonction_algo()
        temps_execution = time.time() - debut
        
        # Traitement du résultat selon le type de retour
        if retourne_noeuds and isinstance(resultat, tuple):
            couverture, noeuds_generes = resultat
            taille_couverture = len(couverture)
        else:
            couverture = resultat
            taille_couverture = len(couverture)
            noeuds_generes = None
        
        # Validation de la couverture
        est_valide = graphe.est_couverture_valide(couverture)
        
        # Affichage des résultats
        print(f"✅ RÉSULTATS - {nom_algo}")
        print(f"   • Temps d'exécution: {temps_execution:.4f} secondes")
        print(f"   • Taille de la couverture: {taille_couverture}")
        print(f"   • Couverture valide: {'✓ OUI' if est_valide else '✗ NON'}")
        
        if noeuds_generes is not None:
            print(f"   • Nœuds générés: {noeuds_generes}")
        
        print(f"   • Couverture: {sorted(couverture)}")
        
        # Calcul de statistiques supplémentaires
        if est_valide:
            ratio_couverture = taille_couverture / n_sommets
            print(f"   • Ratio couverture/sommets: {ratio_couverture:.3f}")
    
    except Exception as e:
        temps_execution = time.time() - debut
        print(f"✗ Erreur lors de l'exécution: {e}")
        print(f"   Temps écoulé: {temps_execution:.4f} secondes")

def tester_tous_algorithmes_approximatifs(graphe):
    """
    Teste tous les algorithmes approximatifs et compare les résultats
    """
    print("\n🔍 COMPARAISON DES ALGORITHMES APPROXIMATIFS")
    print("=" * 50)
    
    algorithmes = [
        ("Algorithme de couplage", graphe.algo_couplage),
        ("Algorithme glouton", graphe.algo_glouton),
        ("Heuristique aléatoire", lambda: graphe.heuristique_aleatoire(20)),
        ("Heuristique degré pondéré", lambda: graphe.heuristique_degre_ponderee(20)),
        ("Heuristique hybride", graphe.heuristique_hybride)
    ]
    
    resultats = []
    
    for nom, algo in algorithmes:
        print(f"\n⏳ Exécution de {nom}...")
        debut = time.time()
        try:
            couverture = algo()
            temps = time.time() - debut
            valide = graphe.est_couverture_valide(couverture)
            resultats.append((nom, len(couverture), temps, valide, couverture))
            print(f"   ✓ Terminé: {len(couverture)} sommets, {temps:.3f}s")
        except Exception as e:
            print(f"   ✗ Erreur: {e}")
            resultats.append((nom, None, None, False, None))
    
    # Affichage comparatif
    print("\n" + "="*60)
    print("RÉSULTATS COMPARATIFS - ALGORITHMES APPROXIMATIFS")
    print("="*60)
    print(f"{'Algorithme':<25} {'Taille':<8} {'Temps (s)':<10} {'Valide':<8}")
    print("-" * 60)
    
    for nom, taille, temps, valide, _ in resultats:
        if taille is not None:
            print(f"{nom:<25} {taille:<8} {temps:<10.4f} {'✓' if valide else '✗':<8}")
        else:
            print(f"{nom:<25} {'ERREUR':<8} {'-':<10} {'✗':<8}")
    
    # Trouver la meilleure solution valide
    solutions_valides = [(nom, taille) for nom, taille, _, valide, _ in resultats 
                        if valide and taille is not None]
    if solutions_valides:
        meilleure_nom, meilleure_taille = min(solutions_valides, key=lambda x: x[1])
        print(f"\n🏆 Meilleure solution: {meilleure_nom} ({meilleure_taille} sommets)")

def tester_tous_algorithmes_branchement(graphe):
    """
    Teste tous les algorithmes de branchement et compare les résultats
    """
    print("\n🔍 COMPARAISON DES ALGORITHMES DE BRANCHEMENT")
    print("=" * 50)
    
    # Vérification de la taille du graphe
    n_sommets = len(graphe.sommets())
    if n_sommets > 20:
        print(f"⚠️  ATTENTION: Le graphe a {n_sommets} sommets.")
        print("   Cette comparaison peut être très longue.")
        reponse = input("   Voulez-vous continuer? (o/n): ").strip().lower()
        if reponse != 'o':
            return
    
    algorithmes = [
        ("Branchement simple", graphe.branchement_simple),
        ("Branchement avec couplage et bornes", graphe.branchement_couplage_avec_borne),
        ("Branchement avec glouton seulement", graphe.branchement_avec_glouton_seulement),
        ("Branchement avec bornes seulement", graphe.branchement_avec_bornes_seulement),
        ("Branchement avec couplage seulement", graphe.branchement_avec_couplage_seulement),
        ("Branchement avec glouton et bornes", graphe.branchement_glouton_avec_borne),
        ("Branchement amélioré v1", graphe.branchement_ameliore_v1),
        ("Branchement amélioré v2", graphe.branchement_ameliore_v2),
        ("Branchement amélioré v3", graphe.branchement_ameliore_v3)
    ]
    
    resultats = []
    
    for nom, algo in algorithmes:
        print(f"\n⏳ Exécution de {nom}...")
        debut = time.time()
        try:
            couverture, noeuds = algo()
            temps = time.time() - debut
            valide = graphe.est_couverture_valide(couverture)
            resultats.append((nom, len(couverture), temps, valide, noeuds, couverture))
            print(f"   ✓ Terminé: {len(couverture)} sommets, {noeuds} nœuds, {temps:.3f}s")
        except Exception as e:
            print(f"   ✗ Erreur: {e}")
            resultats.append((nom, None, None, False, None, None))
    
    # Affichage comparatif
    print("\n" + "="*70)
    print("RÉSULTATS COMPARATIFS - ALGORITHMES DE BRANCHEMENT")
    print("="*70)
    print(f"{'Algorithme':<30} {'Taille':<8} {'Temps (s)':<10} {'Nœuds':<10} {'Valide':<8}")
    print("-" * 70)
    
    for nom, taille, temps, valide, noeuds, _ in resultats:
        if taille is not None:
            noeuds_str = str(noeuds) if noeuds is not None else "N/A"
            print(f"{nom:<30} {taille:<8} {temps:<10.4f} {noeuds_str:<10} {'✓' if valide else '✗':<8}")
        else:
            print(f"{nom:<30} {'ERREUR':<8} {'-':<10} {'-':<10} {'✗':<8}")

def tester_tous_algorithmes(graphe):
    """
    Teste tous les algorithmes disponibles
    """
    print("\n🔍 COMPARAISON COMPLÈTE DE TOUS LES ALGORITHMES")
    print("=" * 50)
    
    # Vérification de la taille pour les algorithmes lents
    n_sommets = len(graphe.sommets())
    if n_sommets > 15:
        print(f"⚠️  ATTENTION: Le graphe a {n_sommets} sommets.")
        print("   Certains algorithmes (branchement, force brute) peuvent être très lents.")
        reponse = input("   Voulez-vous continuer? (o/n): ").strip().lower()
        if reponse != 'o':
            return
    
    # Tester d'abord les algorithmes rapides
    tester_tous_algorithmes_approximatifs(graphe)
    
    # Puis les algorithmes de branchement (si l'utilisateur confirme)
    if n_sommets <= 20:
        print("\n" + "="*50)
        continuer = input("Voulez-vous tester les algorithmes de branchement? (o/n): ").strip().lower()
        if continuer == 'o':
            tester_tous_algorithmes_branchement(graphe)
    
    # Enfin la force brute (seulement pour les très petits graphes)
    if n_sommets <= 15:
        print("\n" + "="*50)
        continuer = input("Voulez-vous tester la force brute? (o/n): ").strip().lower()
        if continuer == 'o':
            print("\n⏳ Exécution de la force brute...")
            debut = time.time()
            try:
                couverture_brute = bruteforce_vertex_cover(graphe.adj)
                temps = time.time() - debut
                valide = graphe.est_couverture_valide(couverture_brute)
                print(f"✓ Force brute terminée: {len(couverture_brute)} sommets, {temps:.3f}s")
                print(f"  Solution optimale: {sorted(couverture_brute)}")
            except Exception as e:
                print(f"✗ Erreur lors de la force brute: {e}")

def main():
    """
    Version du menu principal avec la fonction universelle améliorée
    """
    current_graph = None  # Mémorise le graphe courant
    
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL - ALGORITHMES DE GRAPHES")
        print("="*60)
        if current_graph:
            print(f"📊 Graphe courant: {len(current_graph.sommets())} sommets, {len(current_graph.aretes())} arêtes")
        else:
            print("📊 Aucun graphe chargé")
        print("1. Charger un graphe depuis un fichier")
        print("2. Générer un graphe aléatoire")
        print("3. Afficher les informations du graphe courant")
        print("4. Test des algorithmes GLOUTON et COUPLAGE sur le graphe courant")
        print("5. Test interactif de branchement sur le graphe courant")
        print("6. Comparaison statistique de l'algo GLOUTON vs COUPLAGE")
        print("7. Tests de performance sur plusieurs graphes (benchmark)")
        print("8. Vérification par force brute (petits graphes)")
        print("9. Test des branchements améliorés (v1, v2, v3)")
        print("10. Évaluation de la qualité des algorithmes approximatifs")
        print("11. Évaluation des heuristiques supplémentaires")
        print("12. Tester tous les algorithmes (GRAPHE COURANT)")
        print("13. Quitter")
        print("-"*60)
        
        choix = input("Votre choix (1-13): ").strip()
        
        if choix == "1":
            filename = input("Entrez le nom du fichier (ex: exempleinstance.txt): ").strip()
            try:
                current_graph = Graph(read_graph(filename))
                print(f"✓ Graphe chargé depuis {filename}")
                print(f"  Sommets: {len(current_graph.sommets())}, Arêtes: {len(current_graph.aretes())}")
            except Exception as e:
                print(f"✗ Erreur lors du chargement: {e}")
                
        elif choix == "2":
            # Générer un graphe aléatoire
            try:
                n = int(input("Nombre de sommets (n): "))
                p = float(input("Probabilité des arêtes (p, 0-1): "))
                current_graph = generate_random_graph(n, p)
                print(f"✓ Graphe aléatoire G({n}, {p}) généré")
                print(f"  Sommets: {len(current_graph.sommets())}, Arêtes: {len(current_graph.aretes())}")
            except ValueError as e:
                print(f"✗ Entrée invalide: {e}")
            except Exception as e:
                print(f"✗ Erreur: {e}")
                
        elif choix == "3":
            # Afficher les informations du graphe
            if current_graph is None:
                print("✗ Aucun graphe chargé. Veuillez d'abord charger ou générer un graphe.")
                continue
                
            print("\n--- INFORMATIONS DU GRAPHE ---")
            print(f"Sommets: {current_graph.sommets()}")
            print(f"Arêtes: {current_graph.aretes()}")
            print(f"Degrés par sommet: {current_graph.degrees_dict()}")
            max_verts, max_deg = current_graph.max_degree_vertex(return_all=True)
            print(f"Sommet(s) de degré maximal: {max_verts} (degré {max_deg})")
            
        elif choix == "4":
            # Tester les algorithmes de couverture
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue
                
            print("\n--- ALGORITHMES DE COUVERTURE ---")
            try:
                couverture_couplage = current_graph.algo_couplage()
                couverture_glouton = current_graph.algo_glouton()
                
                print(f"Couverture par couplage: {couverture_couplage}")
                print(f"  Taille: {len(couverture_couplage)}, Valide: {current_graph.est_couverture_valide(couverture_couplage)}")
                
                print(f"Couverture par glouton: {couverture_glouton}")
                print(f"  Taille: {len(couverture_glouton)}, Valide: {current_graph.est_couverture_valide(couverture_glouton)}")
                
            except Exception as e:
                print(f"✗ Erreur lors de l'exécution des algorithmes: {e}")
                
        elif choix == "5":
            # Tester l'algorithme de branchement : simple vs couplage
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue

            print("\n--- ALGORITHME DE BRANCHEMENT : SIMPLE vs COUPLAGE ---")
            print("1. Branchement simple (branchement_simple)")
            print("2. Branchement avec couplage et bornes (branchement_couplage_avec_borne)")
            print("3. Les deux et comparer")
            sous = input("Choix (1-3): ").strip()

            try:
                if len(current_graph.sommets()) > 22:
                    print("⚠️  Attention: le graphe a plus de 22 sommets, le calcul peut être long.")
                    reponse = input("Voulez-vous continuer? (o/n): ").strip().lower()
                    if reponse != 'o':
                        continue

                if sous == "1" or sous == "3":
                    start_time = time.time()
                    res_simple = current_graph.branchement_simple()
                    t_simple = time.time() - start_time
                    cov_simple, nodes_simple = (res_simple if isinstance(res_simple, tuple) else (res_simple, None))

                if sous == "2" or sous == "3":
                    start_time = time.time()
                    res_coupl = current_graph.branchement_couplage_avec_borne()
                    t_coupl = time.time() - start_time
                    cov_coupl, nodes_coupl = (res_coupl if isinstance(res_coupl, tuple) else (res_coupl, None))

                if sous == "1":
                    print(f"\n[Branchement simple] Couverture: {cov_simple}")
                    print(f" Taille: {len(cov_simple)}, Valide: {current_graph.est_couverture_valide(cov_simple)}")
                    print(f" Noeuds générés: {nodes_simple}, Temps: {t_simple:.3f}s")
                elif sous == "2":
                    print(f"\n[Branchement couplage] Couverture: {cov_coupl}")
                    print(f" Taille: {len(cov_coupl)}, Valide: {current_graph.est_couverture_valide(cov_coupl)}")
                    print(f" Noeuds générés: {nodes_coupl}, Temps: {t_coupl:.3f}s")
                else:
                    print("\n--- Résultats comparés ---")
                    print(f"Simple: taille={len(cov_simple)}, noeuds={nodes_simple}, temps={t_simple:.3f}s")
                    print(f"Couplage: taille={len(cov_coupl)}, noeuds={nodes_coupl}, temps={t_coupl:.3f}s")
                    # Comparaison de qualité
                    if len(cov_coupl) < len(cov_simple):
                        print(" => Couplage a trouvé une meilleure solution.")
                    elif len(cov_coupl) > len(cov_simple):
                        print(" => Simple a trouvé une meilleure solution (surprenant).")
                    else:
                        print(" => Même taille de couverture.")
                    # Comparaison de l'efficacité
                    if nodes_simple is not None and nodes_coupl is not None:
                        print(f" => Ratio noeuds (couplage/simple) = {nodes_coupl}/{nodes_simple} = {nodes_coupl/nodes_simple:.3f}")
                    print(f" => Ratio temps (couplage/simple) = {t_coupl:.3f}/{t_simple:.3f} = {t_coupl/t_simple:.3f}")

            except Exception as e:
                print(f"✗ Erreur lors de l'exécution du branchement: {e}")
                
        elif choix == "6":
            # Comparaison des algorithmes glouton et couplage
            print("\n--- COMPARAISON GLouton vs COUPLAGE ---")
            try:
                p = float(input("Probabilité p (défaut 0.3): ") or "0.3")
                num_instances = int(input("Nombre d'instances (défaut 10): ") or "10")
                
                print("Lancement des tests... (cela peut prendre quelques secondes)")
                tests_algos(p=p, num_instances=num_instances)
                
            except ValueError as e:
                print(f"✗ Entrée invalide: {e}")
            except Exception as e:
                print(f"✗ Erreur: {e}")
                
        elif choix == "7":
            # Tests de performance du branchement
            print("\n--- TESTS DE PERFORMANCE DU BRANCHEMENT AVEC DIFFÉRENTES STRATÉGIES ---")
            try:
                # 1. Choix des stratégies à tester
                print("Stratégies disponibles:")
                print("1. Branchement simple (baseline)")
                print("2. Avec couplage et bornes (couplage_borne)")
                print("3. Avec glouton seulement (glouton)")
                print("4. Avec bornes inférieures seulement")
                print("5. Avec couplage seulement (couplage)")
                print("6. Avec glouton et bornes (glouton_borne)")
                print("7. Comparaison: Avec couplage et bornes vs Avec glouton et bornes")
                print("8. Comparaison: Avec algorithme glouton vs Avec couplage seulement")
                print("9. Toutes les stratégies")
                
                strat_choice = input("Choisissez les stratégies à tester (ex: 1,2,3,4,5,6,7,8 ou 9 pour toutes): ").strip()
                if strat_choice == '1':
                    methods_to_test = ['simple']
                elif strat_choice == '2':
                    methods_to_test = ['couplage_borne']
                elif strat_choice == '3':
                    methods_to_test = ['glouton']
                elif strat_choice == '4':
                    methods_to_test = ['bornes']
                elif strat_choice == '5':
                    methods_to_test = ['couplage']
                elif strat_choice == '6':
                    methods_to_test = ['glouton_borne']
                elif strat_choice == '7':
                    methods_to_test = ['couplage_borne', 'glouton_borne']
                    print("Comparaison sélectionnée: Avec couplage et bornes vs Avec glouton et bornes")
                elif strat_choice == '8':
                    methods_to_test = ['glouton', 'couplage']
                    print("Comparaison sélectionnée: Avec algorithme glouton vs Avec couplage seulement")
                else:
                    methods_to_test = ['simple', 'couplage_borne', 'glouton', 'bornes', 'couplage', 'glouton_borne']

                # 2. Configuration des paramètres
                n_values = [8, 10, 12, 14, 16]
                p_values_labels = [0.1, 0.3, 0.5, '1/sqrt']
                num_instances = 3
                max_time_par_instance = 30

                print(f"\nConfiguration actuelle:")
                print(f"  n_values: {n_values}")
                print(f"  p_values: {p_values_labels}")
                print(f"  instances: {num_instances}")
                print(f"  timeout: {max_time_par_instance}s")

                modifier = input("Voulez-vous modifier la configuration? (o/n): ").strip().lower()
                if modifier == 'o':
                    try:
                        n_input = input("n_values (séparés par des virgules): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',') if x.strip()]
                        p_input = input("p_values (séparés par des virgules, ex: 0.1,0.3,0.5,1/sqrt): ")
                        if p_input:
                            p_values_labels = []
                            for p in p_input.split(','):
                                p = p.strip()
                                if p == '1/sqrt':
                                    p_values_labels.append(p)
                                else:
                                    p_values_labels.append(float(p))
                        num_instances = int(input(f"Nombre d'instances (défaut {num_instances}): ") or num_instances)
                        max_time_par_instance = int(input(f"Timeout (défaut {max_time_par_instance}s): ") or max_time_par_instance)
                    except ValueError as e:
                        print(f"✗ Entrée invalide, utilisation des valeurs par défaut: {e}")

                # 3. Lancer les tests
                print("\nLancement des tests... cela peut prendre plusieurs minutes.")
                all_results = tester_strategies_branchement(
                    n_values, p_values_labels,
                    num_instances=num_instances,
                    max_time_par_instance=max_time_par_instance,
                    strategies_to_run=methods_to_test
                )

                # 4. Options d'affichage des résultats
                print("\n" + "="*50)
                print("OPTIONS D'AFFICHAGE DES RÉSULTATS")
                print("="*50)
                print("1. Graphique complet (toutes les stratégies et tous les p)")
                print("2. Graphique par stratégie (chaque stratégie a ses propres sous-graphiques)")
                print("3. Graphique pour une valeur de p spécifique")
                print("4. Graphique comparatif simple (toutes stratégies sur mêmes axes)")
                
                affichage_choice = input("Choisissez le type d'affichage (1-4): ").strip()
                
                if affichage_choice == "1":
                    tracer_comparaison_strategies_complet(all_results, strategies_to_plot=methods_to_test,
                                                        title_suffix=f"(instances={num_instances})")
                elif affichage_choice == "2":
                    tracer_comparaison_strategies_par_strategie(all_results, strategies_to_plot=methods_to_test,
                                                              title_suffix=f"(instances={num_instances})")
                elif affichage_choice == "3":
                    p_choisi = float(input("Entrez la valeur de p à afficher (ex: 0.3): ").strip())
                    tracer_comparaison_strategies_par_p(all_results, strategies_to_plot=methods_to_test,
                                                      p_value=p_choisi, title_suffix=f"(instances={num_instances})")
                elif affichage_choice == "4":
                    tracer_comparaison_strategies_simple(all_results, strategies_to_plot=methods_to_test,
                                                       title_suffix=f"(instances={num_instances})")
                else:
                    print("Option invalide, utilisation de l'affichage par défaut.")
                    tracer_comparaison_strategies_complet(all_results, strategies_to_plot=methods_to_test,
                                                        title_suffix=f"(instances={num_instances})")

            except Exception as e:
                print(f"✗ Erreur lors des tests de performance: {e}")
                
        elif choix == "8":
            # Vérification par force brute
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue
                
            print("\n--- VÉRIFICATION PAR FORCE BRUTE ---")
            try:
                if len(current_graph.sommets()) > 15:  # Réduire la limite pour la sécurité
                    print("✗ La force brute n'est recommandée que pour n ≤ 15")
                    continue
                    
                start_time = time.time()
                couverture_brute = bruteforce_vertex_cover(current_graph.adj)
                temps_ecoule = time.time() - start_time
                
                # Pour debug: trouver toutes les solutions optimales
                all_optimal = bruteforce_vertex_cover_all(current_graph.adj)
                
                print(f"Couverture optimale (force brute): {couverture_brute}")
                print(f"Toutes les solutions optimales: {all_optimal}")
                print(f"Nombre de solutions optimales: {len(all_optimal)}")
                print(f"Taille: {len(couverture_brute)}")
                print(f"Valide: {current_graph.est_couverture_valide(couverture_brute)}")
                print(f"Temps d'exécution: {temps_ecoule:.3f} secondes")
                
            except Exception as e:
                print(f"✗ Erreur lors de la force brute: {e}")
                
        elif choix == "9":
            # Test des branchements améliorés
            print("\n--- TEST DES BRANCHEMENTS AMÉLIORÉS ---")
            try:
                # Configuration des paramètres
                n_values = [8, 10, 12, 14, 16]
                p_values_labels = [0.1, 0.3, 0.5, '1/sqrt']
                num_instances = 3
                max_time_par_instance = 30

                print(f"Configuration actuelle:")
                print(f"  n_values: {n_values}")
                print(f"  p_values: {p_values_labels}")
                print(f"  instances: {num_instances}")
                print(f"  timeout: {max_time_par_instance}s")

                modifier = input("Voulez-vous modifier la configuration? (o/n): ").strip().lower()
                if modifier == 'o':
                    try:
                        n_input = input("n_values (séparés par des virgules): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',') if x.strip()]
                        p_input = input("p_values (séparés par des virgules, ex: 0.1,0.3,0.5,1/sqrt): ")
                        if p_input:
                            p_values_labels = []
                            for p in p_input.split(','):
                                p = p.strip()
                                if p == '1/sqrt':
                                    p_values_labels.append(p)
                                else:
                                    p_values_labels.append(float(p))
                        num_instances = int(input(f"Nombre d'instances (défaut {num_instances}): ") or num_instances)
                        max_time_par_instance = int(input(f"Timeout (défaut {max_time_par_instance}s): ") or max_time_par_instance)
                    except ValueError as e:
                        print(f"✗ Entrée invalide, utilisation des valeurs par défaut: {e}")

                # Lancer les tests
                print("\nLancement des tests des branchements améliorés...")
                all_results, strategy_names = tester_branchement_ameliores(
                    n_values, p_values_labels,
                    num_instances=num_instances,
                    max_time_par_instance=max_time_par_instance
                )

                # Tracer les résultats
                tracer_comparaison_ameliores(all_results, strategy_names,
                                           title_suffix=f"(instances={num_instances})")

            except Exception as e:
                print(f"✗ Erreur lors des tests des branchements améliorés: {e}")
                
        elif choix == "10":
            # Évaluation de la qualité des algorithmes approximatifs
            print("\n--- ÉVALUATION DE LA QUALITÉ DES ALGORITHMES APPROXIMATIFS ---")
            try:
                # Configuration des paramètres
                # Utiliser branchement_ameliore_v3 pour la solution optimale
                n_values = [5, 10, 15, 20, 25]  # Augmenter la limite puisque branchement_ameliore_v3 est plus rapide
                p_values = [0.1, 0.3, 0.5, 0.7]
                num_instances = 10
                max_n_optimal = 100  # Limite pour le calcul optimal avec branchement_ameliore_v3

                print(f"Configuration actuelle:")
                print(f"  n_values: {n_values} (n ≤ {max_n_optimal} pour le calcul des rapports)")
                print(f"  p_values: {p_values}")
                print(f"  instances: {num_instances}")
                print(f"\n✓ Utilisation de branchement_ameliore_v3 pour la solution optimale")
                print("  (plus rapide et garantit l'optimalité)")

                modifier = input("Voulez-vous modifier la configuration? (o/n): ").strip().lower()
                if modifier == 'o':
                    try:
                        n_input = input(f"n_values (séparés par des virgules, n ≤ {max_n_optimal} recommandé): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',') if x.strip()]
                        p_input = input("p_values (séparés par des virgules, ex: 0.1,0.3,0.5,0.7): ")
                        if p_input:
                            p_values = [float(x.strip()) for x in p_input.split(',') if x.strip()]
                        num_instances = int(input(f"Nombre d'instances (défaut {num_instances}): ") or num_instances)
                    except ValueError as e:
                        print(f"✗ Entrée invalide, utilisation des valeurs par défaut: {e}")

                # Lancer l'évaluation
                print("\nLancement de l'évaluation de la qualité des algorithmes approximatifs...")
                print("Utilisation de branchement_ameliore_v3 comme référence optimale.")
                results = evaluer_qualite_approximation(n_values, p_values, num_instances, max_n_optimal)

                # Tracer les résultats
                tracer_rapports_approximation(results, title_suffix=f"(instances={num_instances})", max_n_optimal=max_n_optimal)

            except Exception as e:
                print(f"✗ Erreur lors de l'évaluation de la qualité: {e}")
                import traceback
                traceback.print_exc()
                
        elif choix == "11":
            # Évaluation des heuristiques supplémentaires
            print("\n--- ÉVALUATION DES HEURISTIQUES SUPPLÉMENTAIRES ---")
            try:
                # Configuration des paramètres
                n_values = [10, 20, 30, 40, 50]
                p_values = [0.1, 0.3, 0.5]
                num_instances = 5

                print(f"Configuration actuelle:")
                print(f"  n_values: {n_values}")
                print(f"  p_values: {p_values}")
                print(f"  instances: {num_instances}")

                modifier = input("Voulez-vous modifier la configuration? (o/n): ").strip().lower()
                if modifier == 'o':
                    try:
                        n_input = input("n_values (séparés par des virgules): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',') if x.strip()]
                        p_input = input("p_values (séparés par des virgules, ex: 0.1,0.3,0.5): ")
                        if p_input:
                            p_values = [float(x.strip()) for x in p_input.split(',') if x.strip()]
                        num_instances = int(input(f"Nombre d'instances (défaut {num_instances}): ") or num_instances)
                    except ValueError as e:
                        print(f"✗ Entrée invalide, utilisation des valeurs par défaut: {e}")

                # Lancer l'évaluation
                print("\nLancement de l'évaluation des heuristiques supplémentaires...")
                print("Cette opération peut prendre quelques minutes.")
                results, noms_heuristiques = evaluer_heuristiques(n_values, p_values, num_instances)

                # Tracer les résultats
                tracer_comparaison_heuristiques(results, noms_heuristiques, 
                                              title_suffix=f"(instances={num_instances})")

            except Exception as e:
                print(f"✗ Erreur lors de l'évaluation des heuristiques: {e}")
                
        elif choix == "12":
            current_graph = tester_algorithme_universel_ameliore(current_graph)
            
        elif choix == "13":
            print("Au revoir!")
            break

        else:
            print("✗ Choix invalide. Veuillez choisir un nombre entre 1 et 13.")
        
        # Pause avant de revenir au menu
        if choix not in ["6", "7", "9", "10", "11", "12"]:  # Pas de pause après les tests automatiques
            input("\nAppuyez sur Entrée pour continuer...")
            
if __name__ == "__main__": 
    print("Initialisation...")
    
    # Test avec un graphe chemin
    print("\nTest rapide avec un graphe chemin (0-1-2-3-4):")
    Gpath = Graph({0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]})
    
    print("Algorithme de couplage:", Gpath.algo_couplage())
    print("Algorithme glouton:", Gpath.algo_glouton())
    
    resultat_branchement = Gpath.branchement_simple()
    if isinstance(resultat_branchement, tuple):
        couverture, noeuds = resultat_branchement
        print(f"Branchement simple: {couverture} (noeuds: {noeuds})")
    else:
        print(f"Branchement simple: {resultat_branchement}")
    
    print("Validation:", Gpath.est_couverture_valide(resultat_branchement[0] if isinstance(resultat_branchement, tuple) else resultat_branchement))
    
    # Lancement du menu principal
    main()