# ============================================================================
# Auteurs:
#   - Yuxiang ZHANG (21202829)
#   - Kenan Alsafadi (21502362)
# ============================================================================

import numpy as np
import heapq

def exp(rate):
    """
    Échantillonne à partir d'une distribution exponentielle.
    
    Args:
        rate: Taux (ou liste de taux) de la distribution exponentielle
    
    Returns:
        np.array: Échantillon(s) de la distribution exponentielle
    """
    rate = np.array(rate, dtype=float)
    # Éviter la division par zéro pour les taux nuls
    rate_safe = rate + (rate == 0) * 1e-200
    U = np.random.rand(*rate_safe.shape)
    return -np.log(1 - U) / rate_safe

def simulation(graph, sources, maxT):
    """
    Simule la diffusion d'information dans un graphe selon le modèle de Saito.
    Version optimisée avec file de priorité (heap).
    
    Args:
        graph: Tuple (noms, dict_k, dict_r) représentant le graphe
        sources: Liste des nœuds initialement infectés (temps 0)
        maxT: Temps maximum d'observation
    
    Returns:
        np.array: Temps d'infection de chaque nœud
    """
    names, k_dict, r_dict = graph
    n = len(names)
    
    # 1. Initialisation des temps d'infection
    times = np.full(n, maxT, dtype=float)
    for source in sources:
        times[source] = 0.0
    
    # 2. File de priorité pour les événements à traiter (temps, nœud)
    pq = []
    for source in sources:
        heapq.heappush(pq, (0.0, source))
    
    # 3. Ensemble pour suivre les nœuds déjà traités
    processed = set()
    
    # 4. Construction de la liste des successeurs pour chaque nœud
    succs = {i: [] for i in range(n)}
    for (i, j), k_val in k_dict.items():
        if i < n and j < n:  # Vérification des indices valides
            succs[i].append((j, k_val, r_dict[(i, j)]))
    
    # 5. Boucle principale de simulation
    while pq:
        # 5.1 Extraire le nœud avec le temps minimal
        current_time, current_node = heapq.heappop(pq)
        
        # Ignorer si déjà traité
        if current_node in processed:
            continue
        
        # 5.2 Traiter tous les successeurs du nœud courant
        for j, k_ij, r_ij in succs[current_node]:
            # Vérifier si le successeur peut être infecté
            if times[j] > current_time:
                # Échantillonner selon Bernoulli(k_ij)
                if np.random.rand() < k_ij:
                    # Échantillonner le délai d'infection
                    if r_ij == 0:
                        r_ij_safe = 1e-200  # Éviter division par zéro
                    else:
                        r_ij_safe = r_ij
                    
                    U = np.random.rand()
                    delta = -np.log(1 - U) / r_ij_safe
                    
                    # Calculer le nouveau temps d'infection
                    new_time = current_time + delta
                    
                    # Mettre à jour si meilleur et dans la fenêtre temporelle
                    if new_time < times[j] and new_time < maxT:
                        times[j] = new_time
                        heapq.heappush(pq, (new_time, j))
        
        # 5.3 Marquer le nœud comme traité
        processed.add(current_node)
    
    return times

def getProbaMC(graph, sources, maxT, nbsimu):
    """
    Estime les probabilités marginales d'infection par Monte Carlo.
    
    Args:
        graph: Structure du graphe
        sources: Nœuds sources
        maxT: Temps maximum
        nbsimu: Nombre de simulations
    
    Returns:
        np.array: Probabilités d'infection estimées
    """
    names, _, _ = graph
    n_nodes = len(names)
    counts = np.zeros(n_nodes)
    
    # Simuler nbsimu fois
    for _ in range(nbsimu):
        times = simulation(graph, sources, maxT)
        infected = (times < maxT).astype(int)
        counts += infected
    
    # Calcul des probabilités empiriques
    probabilities = counts / nbsimu
    for s in sources:
        probabilities[s] = 1.0  # Les sources sont toujours infectées
    
    return probabilities

def getPredsSuccs(graph):
    """
    Construit les dictionnaires de prédécesseurs et successeurs.
    
    Args:
        graph: Tuple (noms, dict_k, dict_r)
    
    Returns:
        tuple: (prédécesseurs, successeurs) pour chaque nœud
    """
    names, k_dict, r_dict = graph
    n_nodes = len(names)
    
    preds = {i: [] for i in range(n_nodes)}
    succs = {i: [] for i in range(n_nodes)}
    
    # Parcourir toutes les arêtes
    for (u, v), k_val in k_dict.items():
        r_val = r_dict.get((u, v), 0)
        preds[v].append((u, k_val, r_val))
        succs[u].append((v, k_val, r_val))
    
    return preds, succs

def compute_ab(v, times, preds, maxT, eps=1e-20):
    """
    Calcule les termes a et b pour un nœud v.
    Ces termes sont utilisés dans le calcul de la vraisemblance.
    
    Args:
        v: Nœud cible
        times: Vecteur des temps d'infection
        preds: Dictionnaire des prédécesseurs
        maxT: Temps maximum
        eps: Valeur minimale pour éviter les zéros
    
    Returns:
        tuple: (a_v, b_v) pour le nœud v
    """
    t_v = times[v]
    
    # Cas d'une source (temps d'infection = 0)
    if t_v == 0:
        return 1.0, 0.0
    
    sum_alpha_beta = 0.0
    sum_log_beta = 0.0
    
    # Parcourir tous les prédécesseurs
    for pred, k, r in preds.get(v, []):
        t_pred = times[pred]
        
        # Considérer seulement les prédécesseurs infectés avant v
        if 0 <= t_pred < t_v:
            dt = t_v - t_pred
            beta = k * np.exp(-r * dt) + (1 - k)
            sum_log_beta += np.log(beta)
            
            if beta > eps:
                alpha = k * r * np.exp(-r * dt)
                sum_alpha_beta += alpha / beta
    
    # Calcul de a selon le statut d'infection
    if t_v < maxT:
        a = max(eps, sum_alpha_beta)  # Infecté
    else:
        a = 1.0  # Non infecté
    
    return a, sum_log_beta

def compute_ll(times, preds, maxT):
    """
    Calcule la log-vraisemblance d'une configuration de diffusion.
    
    Args:
        times: Vecteur des temps d'infection
        preds: Dictionnaire des prédécesseurs
        maxT: Temps maximum
    
    Returns:
        tuple: (log-vraisemblance, vecteur a, vecteur b)
    """
    n_nodes = len(times)
    sa = np.zeros(n_nodes)
    sb = np.zeros(n_nodes)
    log_likelihood = 0.0
    
    # Calcul pour chaque nœud
    for v in range(n_nodes):
        a, b = compute_ab(v, times, preds, maxT)
        sa[v] = a
        sb[v] = b
        if a > 0:
            log_likelihood += np.log(a) + b
    
    return log_likelihood, sa, sb

def logsumexp(x):
    """
    Calcule log(∑ exp(x)) de manière numériquement stable.
    
    Args:
        x: Vecteur ou tableau de valeurs
    
    Returns:
        Valeur du log-sum-exp
    """
    x = np.array(x)
    if x.ndim == 1:
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        x_max = np.max(x, axis=-1, keepdims=True)
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=-1))

def addVatT(v, times, newt, preds, succs, sa, sb, maxT, eps=1e-20):
    """
    Ajoute un nœud avec un temps d'infection donné et met à jour sa, sb.
    
    Args:
        v: Nœud à ajouter
        times: Vecteur des temps d'infection
        newt: Nouveau temps d'infection pour v
        preds: Dictionnaire des prédécesseurs
        succs: Dictionnaire des successeurs
        sa: Vecteur a courant
        sb: Vecteur b courant
        maxT: Temps maximum
        eps: Valeur minimale
    """
    # Mettre à jour le temps du nœud
    times[v] = newt
    
    # Recalculer a et b pour le nœud v
    a_v, b_v = compute_ab(v, times, preds, maxT, eps)
    sa[v] = a_v
    sb[v] = b_v
    
    # Mettre à jour les successeurs affectés
    succs_list = succs.get(v, [])
    if len(succs_list) > 0:
        c, k, r = map(np.array, zip(*succs_list))
        tp = times[c]
        which = (tp > newt)  # Successeurs infectés après v
        
        tp = tp[which]
        dt = tp - newt
        k = k[which]
        r = r[which]
        c = c[which]
        
        if len(c) > 0:
            rt = -r * dt
            b1 = k * np.exp(rt)
            b = b1 + 1.0 - k
            a = r * b1
            a = a / b
            b_val = np.log(b)
            
            # Ajouter la contribution aux successeurs
            sa[c] = sa[c] + np.where(tp < maxT, a, 0.0)
            sb[c] = sb[c] + b_val

def removeV(v, times, succs, sa, sb, maxT, eps=1e-20):
    """
    Retire un nœud du réseau et met à jour sa, sb.
    
    Args:
        v: Nœud à retirer
        times: Vecteur des temps d'infection
        succs: Dictionnaire des successeurs
        sa: Vecteur a courant
        sb: Vecteur b courant
        maxT: Temps maximum
        eps: Valeur minimale
    """
    succs_list = succs.get(v, [])
    t = times[v]
    
    if t < 0:  # Déjà retiré
        return
    
    # Marquer comme retiré
    times[v] = -1
    sa[v] = 1.0
    sb[v] = 0.0
    
    if len(succs_list) > 0:
        c, k, r = map(np.array, zip(*succs_list))
        tp = times[c]
        which = (tp > t)  # Successeurs infectés après v
        
        tp = tp[which]
        dt = tp - t
        k = k[which]
        r = r[which]
        c = c[which]
        
        rt = -r * dt
        b1 = k * np.exp(rt)
        b = b1 + 1.0 - k
        
        a = r * b1
        a = a / b
        b_val = np.log(b)
        
        # Retirer la contribution des successeurs
        sa[c] = sa[c] - np.where(tp < maxT, a, 0.0)
        sa[c] = np.where(sa[c] > eps, sa[c], eps)  # Éviter les valeurs négatives
        sb[c] = sb[c] - b_val
        sb[c] = np.where(sb[c] > 0, 0, sb[c])

def getLL(v, times, nt, preds, succs, sa, sb, maxT, onUsers=None):
    """
    Calcule la log-vraisemblance pour un temps d'infection donné d'un nœud.
    
    Args:
        v: Nœud testé
        times: Vecteur des temps d'infection
        nt: Temps d'infection testé pour v
        preds: Dictionnaire des prédécesseurs
        succs: Dictionnaire des successeurs
        sa: Vecteur a courant
        sb: Vecteur b courant
        maxT: Temps maximum
        onUsers: Sous-ensemble de nœuds à considérer
    
    Returns:
        tuple: (log-vraisemblance, nouveaux sa, nouveaux sb)
    """
    sa = np.copy(sa)
    sb = np.copy(sb)
    
    if onUsers is None:
        onUsers = range(len(times))
    
    # Ajouter temporairement le nœud avec le temps testé
    addVatT(v, times, nt, preds, succs, sa, sb, maxT)
    
    # Restaurer l'état original
    times[v] = -1
    
    # Calculer la log-vraisemblance sur les nœuds spécifiés
    ll = np.sum((np.log(sa) + sb)[onUsers])
    return (ll, sa, sb)

def sampleV(v, times, preds, succs, sa, sb, maxT, k, k2):
    """
    Échantillonne un nouveau temps d'infection pour un nœud v.
    
    Args:
        v: Nœud à ré-échantillonner
        times: Vecteur des temps d'infection
        preds: Dictionnaire des prédécesseurs
        succs: Dictionnaire des successeurs
        sa: Vecteur a courant
        sb: Vecteur b courant
        maxT: Temps maximum
        k: Nombre de bins pour l'échantillonnage grossier
        k2: Nombre de points pour l'échantillonnage fin
    """
    # 1. Générer des temps candidats (échantillonnage grossier)
    nbCandidateT = k
    bounds = np.linspace(0, maxT, nbCandidateT)
    newt = np.random.uniform(bounds[:-1], bounds[1:])
    
    # Ajouter le temps courant si le nœud est infecté
    if times[v] < maxT:
        idx = newt.searchsorted(times[v])
        newt = np.concatenate((newt[:idx], [times[v]], newt[idx:]), axis=0)
        nbCandidateT += 1
    
    # Ajouter l'option "non infecté" (temps = maxT)
    newt = np.append(newt, [maxT])
    
    # 2. Déterminer les nœuds affectés par le changement
    succs_list = succs.get(v, [])
    
    # Extraire les indices des successeurs
    c = []
    if succs_list:
        for succ, _, _ in succs_list:
            c.append(succ)
    c.append(v)  # Inclure le nœud v lui-même
    
    c = np.array(c)
    
    # 3. Retirer temporairement le nœud v
    nsa = np.copy(sa)
    nsb = np.copy(sb)
    removeV(v, times, succs, nsa, nsb, maxT)
    
    # 4. Calculer la vraisemblance pour chaque temps candidat
    lls = [getLL(v, times, nt, preds, succs, nsa, nsb, maxT, onUsers=c) for nt in newt]
    ll, la, lb = zip(*lls)
    ll = np.array(ll)
    
    # 5. Calculer les largeurs des bins pour la pondération
    diffsx = (newt[1:] - newt[:-1]) / 2.0
    diffsx[1:] = diffsx[1:] + diffsx[:-1]
    diffsx[0] += newt[0]
    diffsx[-1] += (maxT - newt[nbCandidateT-1]) / 2.0
    
    # 6. Calculer les probabilités (avec stabilisation numérique)
    areas = np.log(diffsx) + ll[:-1]
    lln = np.append(areas, ll[-1])
    
    # Normaliser les probabilités
    p = np.exp(lln - logsumexp(lln))
    
    # 7. Échantillonner un bin (ou maxT)
    i = np.random.choice(range(len(p)), 1, p=p).sum()
    
    if i == (len(p) - 1):  # Cas "non infecté"
        times[v] = maxT
        np.copyto(sa, np.array(la[-1]))
        np.copyto(sb, np.array(lb[-1]))
    else:
        # 8. Échantillonnage fin dans le bin sélectionné
        # Déterminer les bornes du bin
        if i > 0:
            bi = (newt[i] + newt[i-1]) / 2.0
        else:
            bi = 0
            
        if i < (len(p) - 2):
            bs = (newt[i] + newt[i+1]) / 2.0
        else:
            bs = maxT
        
        # Échantillonner k2 points dans le bin
        bounds_fine = np.linspace(bi, bs, k2)
        newt_fine = np.concatenate((
            [newt[i]], 
            np.random.uniform(bounds_fine[:-1], bounds_fine[1:])
        ))
        
        # Calculer la vraisemblance pour chaque point fin
        lls_fine = [getLL(v, times, nt, preds, succs, nsa, nsb, maxT, onUsers=c) 
                   for nt in newt_fine]
        ll_fine, la_fine, lb_fine = zip(*lls_fine)
        ll_fine = np.array(ll_fine)
        
        # Échantillonner selon la vraisemblance
        p_fine = np.exp(ll_fine - logsumexp(ll_fine))
        i_fine = np.random.choice(range(len(p_fine)), 1, p=p_fine).sum()
        
        # Mettre à jour les variables d'état
        times[v] = newt_fine[i_fine]
        np.copyto(sa, np.array(la_fine[i_fine]))
        np.copyto(sb, np.array(lb_fine[i_fine]))

def gb(graph, infections, maxT, sampler, k, k2, burnin=100, ref=None, period=1000):
    """
    Échantillonnage de Gibbs pour estimer les probabilités d'infection.
    
    Args:
        graph: Structure du graphe
        infections: Liste des infections connues [(nœud, temps), ...]
        maxT: Temps maximum
        sampler: Fonction d'échantillonnage (sampleV)
        k: Paramètre d'échantillonnage grossier
        k2: Paramètre d'échantillonnage fin
        burnin: Nombre d'itérations de burn-in
        ref: Probabilités de référence (pour calculer MSE)
        period: Période d'affichage des résultats
    
    Returns:
        np.array: Probabilités d'infection estimées
    """
    names, _, _ = graph
    n = len(names)
    
    # Obtenir les relations de prédécesseurs/successeurs
    preds, succs = getPredsSuccs(graph)
    
    # Initialiser les temps d'infection
    times = np.full(n, maxT, dtype=float)
    for node, t in infections:
        times[node] = t
    
    # Identifier les nœuds variables (non fixés par les observations)
    fixed_nodes = set([node for node, _ in infections])
    variables = [i for i in range(n) if i not in fixed_nodes]
    
    # Initialiser sa et sb
    _, sa, sb = compute_ll(times, preds, maxT)
    
    # Compteur pour les statistiques d'infection
    infection_count = np.zeros(n, dtype=float)
    total_samples = 0
    
    # Boucle principale de Gibbs
    max_epochs = burnin + 10000
    for epoch in range(1, max_epochs + 1):
        # Mélanger l'ordre des variables
        np.random.shuffle(variables)
        
        # Échantillonner chaque variable
        for v in variables:
            sampleV(v, times, preds, succs, sa, sb, maxT, k, k2)
        
        # Collecter des échantillons après le burn-in
        if epoch > burnin:
            total_samples += 1
            infection_count += (times < maxT).astype(float)
            
            # Affichage périodique
            if (epoch - burnin) % period == 0:
                current_rate = infection_count / total_samples
                current_ll, _, _ = compute_ll(times, preds, maxT)
                
                if ref is not None:
                    mse = np.mean((current_rate - ref)**2)
                    print(f"{epoch} {current_rate} MSE = {mse} ll= {current_ll}")
                else:
                    print(f"{epoch} {current_rate} ll= {current_ll}")
    
    # Calcul des probabilités finales
    final_rate = infection_count / total_samples
    return final_rate