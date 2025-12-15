import numpy as np

def exp(rate):
    """Échantillonne des variables selon une loi exponentielle."""
    rate = np.array(rate, dtype=float)
    # Éviter les divisions par 0
    rate_safe = rate + (rate == 0) * 1e-200
    U = np.random.rand(*rate_safe.shape)
    return -np.log(1 - U) / rate_safe

def simulation(graph, sources, maxT):
    """Simule une diffusion unique dans le graphe."""
    names, k_dict, r_dict = graph
    n_nodes = len(names)
    
    # Initialisation
    times = np.full(n_nodes, maxT, dtype=float)
    for s in sources:
        times[s] = 0.0
    
    # Liste des infectieux actifs
    active = [(0.0, s) for s in sources]  # (temps, noeud)
    
    while active:
        # Trouver l'infectieux avec le temps minimal
        active.sort()
        t_i, i = active.pop(0)
        
        if t_i >= maxT:
            continue
            
        # Parcourir tous les nœuds successeurs de i
        for j in range(n_nodes):
            if (i, j) in k_dict:
                k_ij = k_dict[(i, j)]
                r_ij = r_dict[(i, j)]
                
                # Propagation si j n'est pas encore infecté ou infecté plus tard
                if times[j] > t_i:
                    # Échantillonnage de Bernoulli
                    if np.random.random() < k_ij:
                        # Délai de transmission exponentiel
                        delta = -np.log(1 - np.random.random()) / r_ij
                        new_time = t_i + delta
                        
                        # Mettre à jour si le nouveau temps est plus précoce
                        if new_time < times[j]:
                            times[j] = new_time
                            if new_time < maxT:
                                active.append((new_time, j))
    
    return times

def getProbaMC(graph, sources, maxT, nbsimu):
    """Estime les probabilités marginales d'infection par Monte Carlo."""
    names, _, _ = graph
    n_nodes = len(names)
    counts = np.zeros(n_nodes)
    
    for _ in range(nbsimu):
        times = simulation(graph, sources, maxT)
        infected = (times < maxT).astype(int)
        counts += infected
    
    probabilities = counts / nbsimu
    for s in sources:
        probabilities[s] = 1.0
    
    return probabilities

def getPredsSuccs(graph):
    """Retourne les prédécesseurs et successeurs de chaque nœud."""
    names, k_dict, r_dict = graph
    n_nodes = len(names)
    
    preds = {i: [] for i in range(n_nodes)}
    succs = {i: [] for i in range(n_nodes)}
    
    for (u, v), k_val in k_dict.items():
        r_val = r_dict.get((u, v), 0)
        preds[v].append((u, k_val, r_val))
        succs[u].append((v, k_val, r_val))
    
    return preds, succs

def compute_ab(v, times, preds, maxT, eps=1e-20):
    """Calcule les valeurs a et b pour un nœud v."""
    t_v = times[v]
    
    # Si le nœud est une source
    if t_v == 0:
        return 1.0, 0.0
    
    sum_alpha_beta = 0.0
    sum_log_beta = 0.0
    
    for pred, k, r in preds.get(v, []):
        t_pred = times[pred]
        
        # Ne considérer que les prédécesseurs infectés avant v
        if 0 <= t_pred < t_v:
            dt = t_v - t_pred
            beta = k * np.exp(-r * dt) + (1 - k)
            sum_log_beta += np.log(beta)
            
            if beta > eps:
                alpha = k * r * np.exp(-r * dt)
                sum_alpha_beta += alpha / beta
    
    # Calculer la valeur a
    if t_v < maxT:
        a = max(eps, sum_alpha_beta)
    else:
        a = 1.0
    
    return a, sum_log_beta

def compute_ll(times, preds, maxT):
    """Calcule la log-vraisemblance d'une configuration de diffusion."""
    n_nodes = len(times)
    sa = np.zeros(n_nodes)
    sb = np.zeros(n_nodes)
    log_likelihood = 0.0
    
    for v in range(n_nodes):
        a, b = compute_ab(v, times, preds, maxT)
        sa[v] = a
        sb[v] = b
        if a > 0:
            log_likelihood += np.log(a) + b
    
    return log_likelihood, sa, sb

def logsumexp(x):
    """Calcule log(Σ exp(x)) de manière numériquement stable."""
    x = np.array(x)
    if x.ndim == 1:
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        x_max = np.max(x, axis=-1, keepdims=True)
        return x_max + np.log(np.sum(np.exp(x - x_max), axis=-1))
    
def addVatT(v, times, newt, preds, succs, sa, sb, maxT, eps=1e-20):
    """Ajoute un nœud au réseau avec un temps d'infection donné."""
    # Mettre à jour le temps du nœud
    times[v] = newt
    
    # Calculer les nouvelles valeurs a et b pour le nœud v
    a_v, b_v = compute_ab(v, times, preds, maxT, eps)
    sa[v] = a_v
    sb[v] = b_v
    
    # Mettre à jour les successeurs affectés
    succs_list = succs.get(v, [])
    if len(succs_list) > 0:
        c, k, r = map(np.array, zip(*succs_list))
        tp = times[c]
        which = (tp > newt)
        
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
            
            # Ajouter la contribution
            sa[c] = sa[c] + np.where(tp < maxT, a, 0.0)
            sb[c] = sb[c] + b_val
    
def gb(graph, infections, maxT, sampler, burnin=100, ref=None, period=1000, k=10, k2=10):
    """Implémente l'algorithme de Gibbs Sampling."""
    names, _, _ = graph
    n_nodes = len(names)
    
    preds, succs = getPredsSuccs(graph)
    
    # Initialisation des temps
    times = np.full(n_nodes, maxT, dtype=float)
    for node, t in infections:
        times[node] = t
    
    # Variables à échantillonner (non fixées)
    fixed_nodes = [node for node, _ in infections]
    variables = [i for i in range(n_nodes) if i not in fixed_nodes]
    
    # Initialisation des statistiques
    ll, sa, sb = compute_ll(times, preds, maxT)
    
    # Compteurs
    counts = np.zeros(n_nodes)
    n_iter = 0
    
    # Phase de burn-in
    for epoch in range(burnin):
        np.random.shuffle(variables)
        for v in variables:
            sampler(v, times, preds, succs, sa, sb, maxT, k, k2)
    
    # Phase d'échantillonnage
    total_iterations = 10000
    for epoch in range(burnin, total_iterations):
        np.random.shuffle(variables)
        for v in variables:
            sampler(v, times, preds, succs, sa, sb, maxT, k, k2)
        
        # Après le burn-in, accumuler les comptes
        if epoch >= burnin:
            infected = (times < maxT).astype(float)
            counts += infected
            n_iter += 1
            
            # Affichage périodique
            if (epoch - burnin) % period == 0:
                if n_iter > 0:
                    rate = counts / n_iter
                    ll, _, _ = compute_ll(times, preds, maxT)
                    
                    if ref is not None:
                        mse = np.sum(np.power(rate - ref, 2))
                        print(f"{epoch} {rate} MSE = {mse} ll= {ll}")
                    else:
                        print(f"{epoch} {rate} ll= {ll}")
                else:
                    ll, _, _ = compute_ll(times, preds, maxT)
                    print(f"{epoch} burnin time ll= {ll}")
    
    # Calcul final
    if n_iter > 0:
        final_rate = counts / n_iter
    else:
        final_rate = np.zeros(n_nodes)
    
    return final_rate