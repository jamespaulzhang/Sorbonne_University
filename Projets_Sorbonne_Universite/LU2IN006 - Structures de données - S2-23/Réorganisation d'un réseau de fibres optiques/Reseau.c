//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include "Reseau.h"
#include "Chaine.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "SVGwriter.h"

/**
 * @brief Vérifie si un noeud est voisin d'un autre noeud.
 * @param nd Noeud.
 * @param voisin Noeud voisin.
 * @return int Retourne 1 si le noeud est voisin, sinon 0.
 */
int noeudExisteDansVoisins(Noeud* nd, Noeud* voisin){
    CellNoeud* current = nd->voisins;
    while (current != NULL){
        if (current->nd == voisin){
            return 1;
        }
        current = current->suiv;
    }
    return 0;
}

/**
 * @brief Recherche un noeud dans le réseau. S'il n'existe pas, il le crée.
 * @param R Réseau.
 * @param x Coordonnée x du noeud.
 * @param y Coordonnée y du noeud.
 * @return Noeud* Retourne le noeud recherché ou créé.
 */
Noeud* rechercheCreeNoeudListe(Reseau *R, double x, double y){
    CellNoeud *cellule = R->noeuds;
    while (cellule != NULL){
        if (cellule->nd->x == x && cellule->nd->y == y) {
            return cellule->nd;
        }
        cellule = cellule->suiv;
    }

    Noeud *nouveauNoeud = (Noeud*)malloc(sizeof(Noeud));
    if (!nouveauNoeud){
        fprintf(stderr, "Erreur : Allocation mémoire échouée pour Noeud\n");
        exit(1);
    }

    nouveauNoeud->num = R->nbNoeuds + 1;
    nouveauNoeud->x = x;
    nouveauNoeud->y = y;
    nouveauNoeud->voisins = NULL;

    CellNoeud *nouvelleCellule = (CellNoeud*)malloc(sizeof(CellNoeud));
    if (!nouvelleCellule){
        fprintf(stderr, "Erreur : Allocation mémoire échouée pour CellNoeud\n");
        free(nouveauNoeud);
        exit(1);
    }
    nouvelleCellule->nd = nouveauNoeud;
    nouvelleCellule->suiv = R->noeuds;
    R->noeuds = nouvelleCellule;
    R->nbNoeuds++;
    return nouveauNoeud;
}

/**
 * @brief Reconstitue un réseau à partir d'une chaîne en utilisant une liste.
 * @param C Chaîne.
 * @return Reseau* Retourne le réseau reconstitué.
 */
Reseau* reconstitueReseauListe(Chaines* C){
    if (C == NULL || C->nbChaines == 0){
        return NULL;
    }

    Reseau* reseau = (Reseau*)malloc(sizeof(Reseau));
    if (!reseau){
        fprintf(stderr, "Erreur : Allocation mémoire échouée pour le réseau\n");
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
        while (point != NULL){
            Noeud* noeud = rechercheCreeNoeudListe(reseau, point->x, point->y);
            if (noeudPrecedent != NULL && !noeudExisteDansVoisins(noeudPrecedent, noeud)){
                CellNoeud* nouveauVoisin = (CellNoeud*)malloc(sizeof(CellNoeud));
                if (!nouveauVoisin){
                    fprintf(stderr, "Erreur : Allocation mémoire échouée pour CellNoeud\n");
                    libererReseau(reseau);
                    exit(1);
                }
                nouveauVoisin->nd = noeudPrecedent;
                nouveauVoisin->suiv = noeud->voisins;
                noeud->voisins = nouveauVoisin;
                CellNoeud* autreVoisin = (CellNoeud*)malloc(sizeof(CellNoeud));
                if (!autreVoisin){
                    fprintf(stderr, "Erreur : Allocation mémoire échouée pour CellNoeud\n");
                    libererReseau(reseau);
                    exit(1);
                }
                autreVoisin->nd = noeud;
                autreVoisin->suiv = nouveauVoisin->nd->voisins;
                nouveauVoisin->nd->voisins = autreVoisin;
            }
            noeudPrecedent = noeud;
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
        Noeud* noeudDebut = rechercheCreeNoeudListe(reseau, debut->x, debut->y);
        Noeud* noeudFin = rechercheCreeNoeudListe(reseau, fin->x, fin->y);
        CellCommodite* commodite = (CellCommodite*)malloc(sizeof(CellCommodite));
        if (!commodite){
            fprintf(stderr, "Erreur : Allocation mémoire échouée pour CellCommodite\n");
            libererReseau(reseau);
            exit(1);
        }
        commodite->extrA = noeudDebut;
        commodite->extrB = noeudFin;
        commodite->suiv = reseau->commodites;
        reseau->commodites = commodite;
        tempChaine = tempChaine->suiv;
    }
    libererChaines(C);
    return reseau;
}

/**
 * @brief Libère la mémoire allouée pour le réseau.
 * @param R Réseau.
 */
void libererReseau(Reseau* R){
    if (R == NULL) {
        return;
    }
    CellCommodite* currentCommodite = R->commodites;
    while (currentCommodite != NULL){
        CellCommodite* tempCommodite = currentCommodite;
        currentCommodite = currentCommodite->suiv;
        free(tempCommodite);
    }
    CellNoeud* currentNoeud = R->noeuds;
    while (currentNoeud != NULL){
        CellNoeud* tempNoeud = currentNoeud;
        currentNoeud = currentNoeud->suiv;

        CellNoeud* currentVoisin = tempNoeud->nd->voisins;
        while (currentVoisin != NULL){
            CellNoeud* tempVoisin = currentVoisin;
            currentVoisin = currentVoisin->suiv;
            free(tempVoisin);
        }
        free(tempNoeud->nd);
        free(tempNoeud);
    }
    free(R);
}

/**
 * @brief Écrit le réseau dans un fichier.
 * @param R Réseau.
 * @param f Fichier.
 */
void ecrireReseau(Reseau *R, FILE *f){
    fprintf(f, "NbNoeuds: %d\n", R->nbNoeuds);
    fprintf(f, "NbLiaisons: %d\n", nbLiaisons(R));
    fprintf(f, "NbCommodites: %d\n", nbCommodites(R));
    fprintf(f, "Gamma: %d\n\n", R->gamma);
    CellNoeud *current = R->noeuds;
    while (current != NULL){
        fprintf(f, "v %d %f %f\n", current->nd->num, current->nd->x, current->nd->y);
        current = current->suiv;
    }
    fprintf(f, "\n");
    current = R->noeuds;
    while (current != NULL){
        CellNoeud *neighbor = current->nd->voisins;
        while (neighbor != NULL) {
            if (neighbor->nd->num > current->nd->num) {
                fprintf(f, "l %d %d\n", current->nd->num, neighbor->nd->num);
            }
            neighbor = neighbor->suiv;
        }
        current = current->suiv;
    }
    fprintf(f, "\n");
    CellCommodite *currentCommodite = R->commodites;
    while (currentCommodite != NULL){
        fprintf(f, "k %d %d\n", currentCommodite->extrA->num, currentCommodite->extrB->num);
        currentCommodite = currentCommodite->suiv;
    }
    fprintf(f, "\n");
}

/**
 * @brief Compte le nombre de liaisons dans le réseau.
 * @param R Réseau.
 * @return int Nombre de liaisons.
 */
int nbLiaisons(Reseau *R){
    int count = 0;
    CellNoeud *current = R->noeuds;
    while (current != NULL){
        CellNoeud *neighbor = current->nd->voisins;
        while (neighbor != NULL){
            count++;
            neighbor = neighbor->suiv;
        }
        current = current->suiv;
    }
    return count / 2;
}

/**
 * @brief Compte le nombre de commodités dans le réseau.
 * @param R Réseau.
 * @return int Nombre de commodités.
 */
int nbCommodites(Reseau *R){
    int count = 0;
    CellCommodite *current = R->commodites;
    while (current != NULL){
        count++;
        current = current->suiv;
    }
    return count;
}

/**
 * @brief Affiche le réseau sous forme SVG.
 * @param R Réseau.
 * @param nomInstance Nom du fichier SVG.
 */
void afficheReseauSVG(Reseau *R, char* nomInstance){
    CellNoeud *courN,*courv;
    SVGwriter svg;
    double maxx=0,maxy=0,minx=1e6,miny=1e6;
    courN=R->noeuds;
    while (courN!=NULL){
        if (maxx<courN->nd->x) maxx=courN->nd->x;
        if (maxy<courN->nd->y) maxy=courN->nd->y;
        if (minx>courN->nd->x) minx=courN->nd->x;
        if (miny>courN->nd->y) miny=courN->nd->y;
        courN=courN->suiv;
    }
    SVGinit(&svg,nomInstance,500,500);
    courN=R->noeuds;
    while (courN!=NULL){
        SVGpoint(&svg,500*(courN->nd->x-minx)/(maxx-minx),500*(courN->nd->y-miny)/(maxy-miny));
        courv=courN->nd->voisins;
        while (courv!=NULL){
            if (courv->nd->num<courN->nd->num)
                SVGline(&svg,500*(courv->nd->x-minx)/(maxx-minx),500*(courv->nd->y-miny)/(maxy-miny),500*(courN->nd->x-minx)/(maxx-minx),500*(courN->nd->y-miny)/(maxy-miny));
            courv=courv->suiv;
        }
        courN=courN->suiv;
    }
    SVGfinalize(&svg);
}
