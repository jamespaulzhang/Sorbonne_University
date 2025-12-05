//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Hachage.h"

/**
 * @brief Calcule la clé pour un point (x, y).
 * @param x Coordonnée x du point.
 * @param y Coordonnée y du point.
 * @return unsigned int Clé calculée pour le point.
 */
unsigned int clef(double x, double y){
    return (unsigned int)(y + (x + y) * (x + y + 1) / 2);
}

/**
 * @brief Fonction de hachage.
 * @param x Coordonnée x du point.
 * @param y Coordonnée y du point.
 * @param tailleMax Taille maximale de la table de hachage.
 * @return unsigned int Valeur de hachage pour le point (x, y).
 */
unsigned int fonctionHachage(double x, double y, int tailleMax){
    const double A = (sqrt(5.0) - 1.0) / 2.0;
    unsigned int k = clef(x, y);
    return (unsigned int)floor(tailleMax * (k * A - floor(k * A)));
}

/**
 * @brief Crée une nouvelle table de hachage.
 * @param tailleMax Taille maximale de la table de hachage.
 * @return TableHachage* Pointeur vers la table de hachage créée.
 */
TableHachage* creerTableHachage(int tailleMax){
    TableHachage* table = (TableHachage*)malloc(sizeof(TableHachage));
    if (!table){
        fprintf(stderr, "Erreur d'allocation de mémoire pour la table de hachage\n");
        exit(1);
    }
    table->tailleMax = tailleMax;
    table->T = (CellNoeud**)calloc(tailleMax, sizeof(CellNoeud*));
    if (!table->T){
        free(table);
        return NULL;
    }
    return table;
}

/**
 * @brief Libère la mémoire allouée pour la table de hachage.
 * @param H Table de hachage à libérer.
 */
void detruireTableHachage(TableHachage* H){
    if (H){
        for (int i = 0; i < H->tailleMax; i++){
            CellNoeud* current = H->T[i];
            while (current != NULL){
                CellNoeud* temp = current;
                current = current->suiv;
                free(temp);
            }
        }
        free(H->T);
        free(H);
    }
}

/**
 * @brief Ajoute un voisin à un nœud.
 * @param noeud Nœud auquel ajouter le voisin.
 * @param voisin Nœud voisin à ajouter.
 */
void ajouterVoisin(Noeud* noeud, Noeud* voisin){
    CellNoeud* nouvelleCellule = (CellNoeud*)malloc(sizeof(CellNoeud));
    if (!nouvelleCellule){
        fprintf(stderr, "Erreur d'allocation de mémoire pour CellNoeud\n");
        exit(1);
    }
    nouvelleCellule->nd = voisin;
    nouvelleCellule->suiv = noeud->voisins;
    noeud->voisins = nouvelleCellule;
}

/**
 * @brief Ajoute une commodité au réseau.
 * @param reseau Réseau auquel ajouter la commodité.
 * @param extrA Extrémité A de la commodité.
 * @param extrB Extrémité B de la commodité.
 */
void ajouterCommodite(Reseau* reseau, Noeud* extrA, Noeud* extrB){
    CellCommodite* nouvelleCommodite = (CellCommodite*)malloc(sizeof(CellCommodite));
    if (!nouvelleCommodite){
        fprintf(stderr, "Erreur d'allocation de mémoire pour CellCommodite\n");
        exit(1);
    }
    nouvelleCommodite->extrA = extrA;
    nouvelleCommodite->extrB = extrB;
    nouvelleCommodite->suiv = reseau->commodites;
    reseau->commodites = nouvelleCommodite;
}

int numNoeud = 1;

/**
 * @brief Recherche ou crée un nœud dans la table de hachage.
 * @param R Réseau auquel appartient le nœud.
 * @param H Table de hachage dans laquelle rechercher ou ajouter le nœud.
 * @param x Coordonnée x du nœud.
 * @param y Coordonnée y du nœud.
 * @return Noeud* Pointeur vers le nœud recherché ou créé.
 */
Noeud* rechercheCreeNoeudHachage(Reseau* R, TableHachage* H, double x, double y){
    unsigned int cle = fonctionHachage(x, y, H->tailleMax);
    CellNoeud* liste = H->T[cle];
    while (liste != NULL){
        if (liste->nd->x == x && liste->nd->y == y){
            return liste->nd;
        }
        liste = liste->suiv;
    }

    Noeud* nouveauNoeud = (Noeud*)malloc(sizeof(Noeud));
    if (!nouveauNoeud){
        fprintf(stderr, "Erreur d'allocation de mémoire pour le nouveau nœud\n");
        exit(1);
    }
    nouveauNoeud->num = numNoeud++;
    nouveauNoeud->x = x;
    nouveauNoeud->y = y;
    nouveauNoeud->voisins = NULL;

    CellNoeud* celluleReseau = (CellNoeud*)malloc(sizeof(CellNoeud));
    if (!celluleReseau){
        fprintf(stderr, "Erreur d'allocation de mémoire pour la nouvelle cellule du réseau\n");
        free(nouveauNoeud);
        exit(1);
    }
    celluleReseau->nd = nouveauNoeud;
    celluleReseau->suiv = R->noeuds;
    R->noeuds = celluleReseau;
    R->nbNoeuds++;

    CellNoeud* celluleHash = (CellNoeud*)malloc(sizeof(CellNoeud));
    if (!celluleHash){
        fprintf(stderr, "Erreur d'allocation de mémoire pour la nouvelle cellule de la table de hachage\n");
        free(nouveauNoeud);
        free(celluleReseau);
        exit(1);
    }
    celluleHash->nd = nouveauNoeud;
    celluleHash->suiv = H->T[cle];
    H->T[cle] = celluleHash;
    return nouveauNoeud;
}

/**
 * @brief Reconstruit un réseau à partir d'une chaîne de caractères.
 * @param C Chaînes de caractères contenant les données du réseau.
 * @param M Taille maximale de la table de hachage.
 * @return Reseau* Réseau reconstruit à partir des données.
 */
Reseau* reconstitueReseauHachage(Chaines* C, int M){
    TableHachage* H = creerTableHachage(M);
    Reseau* reseau = (Reseau*)malloc(sizeof(Reseau));
    if (!reseau){
        fprintf(stderr, "Erreur d'allocation de mémoire pour le réseau\n");
        exit(1);
    }
    reseau->nbNoeuds = 0;
    reseau->gamma = C->gamma;
    reseau->noeuds = NULL;
    reseau->commodites = NULL;
    CellChaine* chaine = C->chaines;
    while (chaine != NULL){
        CellPoint* point = chaine->points;
        Noeud* noeudPrecedent = NULL;
        Noeud* noeudActuel = NULL;
        while (point != NULL){
            noeudActuel = rechercheCreeNoeudHachage(reseau, H, point->x, point->y);
            if (noeudPrecedent != NULL && !noeudExisteDansVoisins(noeudPrecedent, noeudActuel)){
                ajouterVoisin(noeudActuel, noeudPrecedent);
                ajouterVoisin(noeudPrecedent, noeudActuel);
            }
            noeudPrecedent = noeudActuel;
            point = point->suiv;
        }
        chaine = chaine->suiv;
    }
    CellChaine* tempChaine = C->chaines;
    while (tempChaine != NULL){
        CellPoint* debut = tempChaine->points;
        CellPoint* fin = tempChaine->points;
        while (fin->suiv != NULL){
            fin = fin->suiv;
        }
        Noeud* noeudDebut = rechercheCreeNoeudHachage(reseau, H, debut->x, debut->y);
        Noeud* noeudFin = rechercheCreeNoeudHachage(reseau, H, fin->x, fin->y);
        ajouterCommodite(reseau, noeudDebut, noeudFin);
        tempChaine = tempChaine->suiv;
    }
    libererChaines(C);
    detruireTableHachage(H);
    return reseau;
}
