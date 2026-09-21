//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include "ArbreQuat.h"
#include "Hachage.h"
#include <stdio.h>
#include <stdlib.h>

int global_node_id = 1;

/**
 * @brief Détermine les coordonnées minimales et maximales des points dans les chaînes.
 * @param C Chaînes dont on veut trouver les coordonnées minimales et maximales.
 * @param xmin Pointeur vers la variable où stocker la coordonnée minimale en x.
 * @param ymin Pointeur vers la variable où stocker la coordonnée minimale en y.
 * @param xmax Pointeur vers la variable où stocker la coordonnée maximale en x.
 * @param ymax Pointeur vers la variable où stocker la coordonnée maximale en y.
 */
void chaineCoordMinMax(Chaines* C, double* xmin, double* ymin, double* xmax, double* ymax){
    if (C == NULL || C->chaines == NULL){
        printf("Liste de Chaines vide.\n");
        return;
    }
    *xmin = C->chaines->points->x;
    *xmax = C->chaines->points->x;
    *ymin = C->chaines->points->y;
    *ymax = C->chaines->points->y;
    CellChaine* chaine = C->chaines;
    while (chaine != NULL){
        CellPoint* point = chaine->points;
        while (point != NULL){
            if (point->x < *xmin){
                *xmin = point->x;
            }
            if (point->x > *xmax){
                *xmax = point->x;
            }
            if (point->y < *ymin){
                *ymin = point->y;
            }
            if (point->y > *ymax){
                *ymax = point->y;
            }
            point = point->suiv;
        }
        chaine = chaine->suiv;
    }
}

/**
 * @brief Crée un arbre quaternaire avec les coordonnées et les dimensions spécifiées.
 * @param xc Coordonnée en x du centre de l'arbre.
 * @param yc Coordonnée en y du centre de l'arbre.
 * @param coteX Longueur du côté en x de l'arbre.
 * @param coteY Longueur du côté en y de l'arbre.
 * @return ArbreQuat* Pointeur vers le nouvel arbre quaternaire créé.
 */
ArbreQuat* creerArbreQuat(double xc, double yc, double coteX, double coteY){
    ArbreQuat* arbre = (ArbreQuat*)malloc(sizeof(ArbreQuat));
    if (arbre == NULL){
        fprintf(stderr, "Erreur d'allocation de mémoire pour l'arbre quaternaire\n");
        exit(1);
    }
    arbre->xc = xc;
    arbre->yc = yc;
    arbre->coteX = coteX;
    arbre->coteY = coteY;
    arbre->noeud = NULL;
    arbre->so = NULL;
    arbre->se = NULL;
    arbre->no = NULL;
    arbre->ne = NULL;
    return arbre;
}

/**
 * @brief Libère la mémoire allouée pour un arbre quaternaire.
 * @param a Arbre quaternaire à libérer.
 */
void libererArbreQuat(ArbreQuat* a){
    if (a == NULL) {
        return;
    }
    a->noeud = NULL;
    libererArbreQuat(a->no);
    libererArbreQuat(a->ne);
    libererArbreQuat(a->so);
    libererArbreQuat(a->se);
    free(a);
}

/**
 * @brief Détermine le quadrant dans lequel se trouve un point par rapport à un nœud parent.
 * @param x Coordonnée en x du point.
 * @param y Coordonnée en y du point.
 * @param parent Nœud parent.
 * @return char Quadrant dans lequel se trouve le point ('1', '2', '3' ou '4').
 */
char trouverQuadrant(double x, double y, ArbreQuat* parent){
    if (x < parent->xc){
        if (y > parent->yc){
            return '1'; // nord-ouest
        } else {
            return '2'; // sud-ouest
        }
    } else {
        if (y > parent->yc){
            return '3'; // nord-est
        } else {
            return '4'; // sud-est
        }
    }
}

/**
 * @brief Insère un nœud dans un arbre quaternaire.
 * @param n Nœud à insérer.
 * @param a Adresse du pointeur vers l'arbre quaternaire.
 * @param parent Nœud parent.
 */
void insererNoeudArbre(Noeud* n, ArbreQuat** a, ArbreQuat* parent){
    if (*a == NULL){
        double xc, yc, coteX, coteY;
        xc = (n->x < parent->xc) ? parent->xc - parent->coteX / 4 : parent->xc + parent->coteX / 4;
        yc = (n->y < parent->yc) ? parent->yc - parent->coteY / 4 : parent->yc + parent->coteY / 4;
        coteX = parent->coteX / 2;
        coteY = parent->coteY / 2;
        *a = creerArbreQuat(xc, yc, coteX, coteY);
        (*a)->noeud = n;
        return;
    }
    if ((*a)->noeud != NULL){
        Noeud* ancienNoeud = (*a)->noeud;
        (*a)->noeud = NULL; 
        insererNoeudArbre(ancienNoeud, a, parent);
        insererNoeudArbre(n, a, parent);
        return;
    }
    if ((*a)->noeud == NULL){
        ArbreQuat** sousArbre = NULL;
        if (n->x < (*a)->xc) {
            sousArbre = (n->y < (*a)->yc) ? &((*a)->so) : &((*a)->no);
        } else {
            sousArbre = (n->y < (*a)->yc) ? &((*a)->se) : &((*a)->ne);
        }
        insererNoeudArbre(n, sousArbre, *a);
    }
}

/**
 * @brief Recherche un nœud par ses coordonnées dans un réseau.
 * @param R Réseau dans lequel effectuer la recherche.
 * @param x Coordonnée en x du nœud recherché.
 * @param y Coordonnée en y du nœud recherché.
 * @return Noeud* Pointeur vers le nœud trouvé, NULL si aucun nœud trouvé.
 */
Noeud* rechercheNoeudParCoordonnees(Reseau* R, double x, double y){
    if (R == NULL || R->noeuds == NULL){
        return NULL;
    }
    CellNoeud* noeud = R->noeuds;
    while (noeud != NULL){
        if (noeud->nd != NULL && noeud->nd->x == x && noeud->nd->y == y){
            return noeud->nd;
        }
        noeud = noeud->suiv;
    }

    return NULL;
}

/**
 * @brief Recherche un nœud par ses coordonnées dans un arbre quaternaire. Si le nœud n'existe pas, il est créé.
 * @param R Réseau dans lequel effectuer la recherche.
 * @param a Adresse du pointeur vers l'arbre quaternaire.
 * @param parent Nœud parent de l'arbre quaternaire.
 * @param x Coordonnée en x du nœud recherché.
 * @param y Coordonnée en y du nœud recherché.
 * @return Noeud* Pointeur vers le nœud trouvé ou créé.
 */
Noeud* rechercheCreeNoeudArbre(Reseau* R, ArbreQuat** a, ArbreQuat* parent, double x, double y){
    Noeud* noeudExistant = rechercheNoeudParCoordonnees(R, x, y);
    if (noeudExistant != NULL){
        return noeudExistant;
    }
    if (*a == NULL){
        Noeud* nouveauNoeud = (Noeud*)malloc(sizeof(Noeud));
        if (nouveauNoeud == NULL){
            fprintf(stderr, "Erreur : Échec de l'allocation de mémoire pour le nouveau nœud.\n");
            return NULL;
        }
        nouveauNoeud->x = x;
        nouveauNoeud->y = y;
        nouveauNoeud->num = global_node_id++;
        nouveauNoeud->voisins = NULL;
        insererNoeudArbre(nouveauNoeud, a, parent);
        CellNoeud* celluleNoeud = (CellNoeud*)malloc(sizeof(CellNoeud));
        if (celluleNoeud == NULL){
            fprintf(stderr, "Erreur : Échec de l'allocation de mémoire pour la cellule de nœud.\n");
            free(nouveauNoeud);
            return NULL;
        }
        celluleNoeud->nd = nouveauNoeud;
        celluleNoeud->suiv = R->noeuds;
        R->noeuds = celluleNoeud;
        R->nbNoeuds++;
        return nouveauNoeud;
    }else{
        char quadrant = trouverQuadrant(x, y, *a);
        ArbreQuat** sousArbre = NULL;
        switch (quadrant){
            case '1':
                sousArbre = &((*a)->no);
                break;
            case '2':
                sousArbre = &((*a)->so);
                break;
            case '3':
                sousArbre = &((*a)->ne);
                break;
            case '4':
                sousArbre = &((*a)->se);
                break;
        }
        return rechercheCreeNoeudArbre(R, sousArbre, *a, x, y);
    }
}

/**
 * @brief Reconstruit un réseau à partir d'une liste de chaînes.
 * @param C Liste des chaînes.
 * @return Reseau* Réseau reconstruit.
 */
Reseau* reconstitueReseauArbre(Chaines* C){
    if (C->nbChaines < 0){
        fprintf(stderr, "Error: Invalid chains\n");
        return NULL;
    }
    if (C == NULL || C->nbChaines == 0){
        return NULL;
    }
    double xmin, ymin, xmax, ymax;
    chaineCoordMinMax(C, &xmin, &ymin, &xmax, &ymax);
    ArbreQuat* arbre = creerArbreQuat((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin);
    Reseau* reseau = (Reseau*)malloc(sizeof(Reseau));
    if (!reseau) {
        fprintf(stderr, "Erreur d'allocation de mémoire pour le réseau\n");
        libererArbreQuat(arbre);
        return NULL;
    }
    reseau->nbNoeuds = 0;
    reseau->gamma = C->gamma;
    reseau->noeuds = NULL;
    reseau->commodites = NULL;
    CellChaine* chaine = C->chaines;
    while (chaine != NULL){
        CellPoint* point = chaine->points;
        Noeud* premierNoeud = NULL;
        Noeud* dernierNoeud = NULL;
        Noeud* noeudPrecedent = NULL;
        while (point != NULL){
            Noeud* noeud = rechercheCreeNoeudArbre(reseau, &arbre, NULL, point->x, point->y);
            if (premierNoeud == NULL){
                premierNoeud = noeud;
            }
            dernierNoeud = noeud;
            if (noeudPrecedent != NULL && !noeudExisteDansVoisins(noeudPrecedent, noeud)){
                ajouterVoisin(noeudPrecedent, noeud);
                ajouterVoisin(noeud, noeudPrecedent);
            }
            noeudPrecedent = noeud;
            point = point->suiv;
        }
        if (premierNoeud != NULL && dernierNoeud != NULL ) {
            ajouterCommodite(reseau, premierNoeud, dernierNoeud);
        }
        chaine = chaine->suiv;
    }
    libererChaines(C);
    libererArbreQuat(arbre);
    return reseau;
}
