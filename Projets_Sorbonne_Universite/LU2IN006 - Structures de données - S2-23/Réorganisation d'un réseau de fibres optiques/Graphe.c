//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include "Graphe.h"
#include "ArbreQuat.h"
#include "Chaine.h"
#include "Struct_File.h"
#include "Hachage.h"
#include "Reseau.h"

/**
 * @brief Crée un graphe à partir d'un réseau.
 * @param r Réseau à partir duquel créer le graphe.
 * @return Graphe* Pointeur vers le graphe créé.
 */
Graphe* creerGraphe(Reseau* r){
    Graphe* g = (Graphe*)malloc(sizeof(Graphe));
    g->nbsom = r->nbNoeuds;
    g->gamma = r->gamma;
    g->nbcommod = nbCommodites(r);
    g->T_commod = (Commod*)malloc(g->nbcommod * sizeof(Commod));
    g->T_som = (Sommet**)malloc((g->nbsom + 1) * sizeof(Sommet*));
    CellNoeud* n = r->noeuds;
    CellCommodite* c = r->commodites;
    int i = 0;
    while (c != NULL){
        g->T_commod[i].e1 = c->extrA->num;
        g->T_commod[i].e2 = c->extrB->num;
        i++;
        c = c->suiv;
    }
    while (n != NULL){
        Sommet* s = (Sommet*)malloc(sizeof(Sommet));
        s->num = n->nd->num;
        s->x = n->nd->x;
        s->y = n->nd->y;
        s->L_voisin = NULL;
        CellNoeud* voisins = n->nd->voisins;
        while (voisins != NULL) {
            Arete* a = (Arete*)malloc(sizeof(Arete));
            a->u = n->nd->num;
            a->v = voisins->nd->num;
            Cellule_arete* ca = (Cellule_arete*)malloc(sizeof(Cellule_arete));
            ca->a = a;
            ca->suiv = s->L_voisin;
            s->L_voisin = ca;
            voisins = voisins->suiv;
        }
        g->T_som[n->nd->num] = s;
        n = n->suiv;
    }
    return g;
}

/**
 * @brief Calcule la longueur minimale du chemin entre deux sommets dans un graphe.
 * @param g Graphe dans lequel rechercher le chemin.
 * @param u Sommet de départ.
 * @param v Sommet d'arrivée.
 * @return int Longueur minimale du chemin entre u et v, -1 si aucun chemin n'existe.
 */
int MinNbAretes(Graphe* g, int u, int v){
    S_file* file = (S_file*)malloc(sizeof(S_file));
    Init_file(file);
    int* visite = (int*)malloc((g->nbsom + 1) * sizeof(int));
    for (int i = 1; i < g->nbsom + 1; i++){
        visite[i] = 0;
    }
    visite[u] = 1;
    enfile(file, u); 

    while (!estFileVide(file)){
        int s = defile(file);
        if (s == v){
            int result = visite[s] - 1;
            free(visite);
            libererFile(file);
            return result;
        }
        else{ 
            Cellule_arete* cell = g->T_som[s]->L_voisin;
            while (cell != NULL) {
                if (visite[cell->a->v] == 0){
                    visite[cell->a->v] = visite[cell->a->u] + 1;
                    enfile(file, cell->a->v);
                }
                cell = cell->suiv;
            }
        }
    }
    free(visite);
    libererFile(file);
    return -1;
}

/**
 * @brief Calcule le chemin le plus court entre deux sommets dans un graphe.
 * @param g Graphe dans lequel rechercher le chemin.
 * @param u Sommet de départ.
 * @param v Sommet d'arrivée.
 * @return S_file* File représentant le chemin le plus court entre u et v.
 */
S_file* MinListeEntiers(Graphe* g, int u, int v){
    S_file* file = (S_file*)malloc(sizeof(S_file));
    Init_file(file);
    int* visite = (int*)malloc((g->nbsom + 1) * sizeof(int));
    int* predecesseurs = (int*)malloc((g->nbsom + 1) * sizeof(int));
    for (int i = 0; i < g->nbsom + 1; i++){
        visite[i] = 0;
        predecesseurs[i] = 0;
    }
    visite[u] = 1;
    enfile(file, u);
    predecesseurs[u] = 0;
    while (!estFileVide(file)){
        int s = defile(file);
        if (s == v) {
            S_file* chaine = (S_file *)malloc(sizeof(S_file));
            Init_file(chaine);
            int x = v;
            printf("Chemin trouvé : ");
            while (x != 0) {
                printf("%d ", x);
                enfile(chaine, x);
                x = predecesseurs[x];
            }
            printf("\n");

            S_file* chaine_finale = (S_file*)malloc(sizeof(S_file));
            Init_file(chaine_finale);
            while (!estFileVide(chaine)){
                enfile(chaine_finale, defile(chaine));
            }
            libererFile(chaine);
            free(visite);
            free(predecesseurs);
            libererFile(file);
            return chaine_finale;
        }
        else{
            Cellule_arete* cell = g->T_som[s]->L_voisin;
            while (cell != NULL){
                if (visite[cell->a->v] == 0){
                    visite[cell->a->v] = visite[cell->a->u] + 1;
                    predecesseurs[cell->a->v] = cell->a->u;
                    enfile(file, cell->a->v);
                }
                cell = cell->suiv;
            }
        }
    }
    free(visite);
    free(predecesseurs);
    libererFile(file);
    return NULL;
}

/**
 * @brief Réorganise un réseau pour que chaque arête supporte moins de chaînes que gamma.
 * @param r Réseau à réorganiser.
 * @return int Renvoie 1 si le réseau peut être réorganisé, sinon 0.
 */
int reorganiseReseau(Reseau* r){
    Graphe* g = creerGraphe(r);
    int** matrice = (int**)malloc((g->nbsom + 1) * sizeof(int*));
    for (int i = 0; i <= g->nbsom; i++){
        matrice[i] = (int *)malloc((g->nbsom + 1) * sizeof(int));
        for (int j = 0; j <= g->nbsom; j++){
            matrice[i][j] = 0;
        }
    }
    int* TminTaille = (int*)malloc(g->nbcommod * sizeof(int));
    S_file** TminChaine = (S_file**)malloc(g->nbcommod * sizeof(S_file*));
    for (int k = 0; k < g->nbcommod; k++){
        TminTaille[k] = MinNbAretes(g, g->T_commod[k].e1, g->T_commod[k].e2);
        TminChaine[k] = MinListeEntiers(g, g->T_commod[k].e1, g->T_commod[k].e2);
    }
    int res = verifierNbChaines(g, matrice);
    for (int i = 0; i < g->nbcommod; i++){
        libererFile(TminChaine[i]);
    }
    free(TminTaille);
    free(TminChaine);
    for (int i = 0; i <= g->nbsom; i++){
        free(matrice[i]);
    }
    free(matrice);
    libererGraphe(g);
    return res;
}

/**
 * @brief Vérifie si le nombre de chaînes passant par chaque arête du graphe est inférieur à gamma.
 * @param g Graphe représenté par une liste d'adjacence.
 * @param matrice Matrice d'adjacence indiquant les arêtes du graphe.
 * @return int Renvoie 1 si pour toute arête du graphe, le nombre de chaînes qui passe par cette arête est strictement inférieur à gamma, sinon 0.
 */
int verifierNbChaines(Graphe* g, int** matrice){
    for (int i = 1; i <= g->nbsom; i++) {
        for (Cellule_arete* voisin = g->T_som[i]->L_voisin; voisin != NULL; voisin = voisin->suiv) {
            int u = i;
            int v = voisin->a->v;
            if (matrice[u][v] >= g->gamma){
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Libère la mémoire allouée pour le graphe.
 * @param g Graphe à libérer.
 */
void libererGraphe(Graphe* g){
    for (int i = 1; i <= g->nbsom; i++){
        Cellule_arete* courant = g->T_som[i]->L_voisin;
        Cellule_arete* suivant;
        while (courant != NULL) {
            suivant = courant->suiv;
            free(courant->a);
            free(courant);
            courant = suivant;
        }
        free(g->T_som[i]);
    }
    free(g->T_som);
    free(g->T_commod);
    free(g);
}

/**
 * @brief Libère la mémoire allouée pour une file.
 * @param f File à libérer.
 */
void libererFile(S_file* f){
    Cellule_file* courant = f->tete;
    Cellule_file* suivant;
    while (courant != NULL){
        suivant = courant->suiv;
        free(courant);
        courant = suivant;
    }
    free(f);
}
