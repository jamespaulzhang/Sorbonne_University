//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "Chaine.h"
#include "SVGwriter.h" 

/**
 * @brief Lit les chaînes à partir d'un fichier.
 * @param f Pointeur vers le fichier à partir duquel lire les chaînes.
 * @return Chaines* Chaînes lues à partir du fichier.
 */
Chaines* lectureChaines(FILE *f){
    Chaines *chaines = (Chaines*)malloc(sizeof(Chaines));
    if (chaines == NULL){
        fprintf(stderr, "Erreur d'allocation de mémoire pour Chaines.\n");
        return NULL;
    }
    chaines->gamma = 0;
    chaines->nbChaines = 0;
    chaines->chaines = NULL;
    int nbChaines, gamma;
    if (fscanf(f, "NbChain: %d\nGamma: %d\n", &nbChaines, &gamma) != 2){
        fprintf(stderr, "Erreur de lecture du nombre de chaînes et du nombre maximal de fibres par câble.\n");
        free(chaines);
        return NULL;
    }
    chaines->gamma = gamma;
    chaines->nbChaines = nbChaines;
    for (int i = 0; i < nbChaines; ++i){
        CellChaine *nouvelleChaine = (CellChaine*)malloc(sizeof(CellChaine));
        if (nouvelleChaine == NULL) {
            fprintf(stderr, "Erreur d'allocation de mémoire pour CellChaine.\n");
            libererChaines(chaines);
            return NULL;
        }
        nouvelleChaine->points = NULL;
        if (fscanf(f, "%d", &(nouvelleChaine->numero)) != 1){
            fprintf(stderr, "Erreur de lecture du numéro de la chaîne.\n");
            free(nouvelleChaine);
            libererChaines(chaines);
            return NULL;
        }
        int nbPoints;
        if (fscanf(f, "%d", &nbPoints) != 1){
            fprintf(stderr, "Erreur de lecture du nombre de points de la chaîne.\n");
            free(nouvelleChaine);
            libererChaines(chaines);
            return NULL;
        }
        for (int j = 0; j < nbPoints; ++j){
            CellPoint *nouveauPoint = (CellPoint*)malloc(sizeof(CellPoint));
            if (nouveauPoint == NULL) {
                fprintf(stderr, "Erreur d'allocation de mémoire pour CellPoint.\n");
                libererChaines(chaines);
                return NULL;
            }
            if (fscanf(f, "%lf %lf", &(nouveauPoint->x), &(nouveauPoint->y)) != 2){
                fprintf(stderr, "Erreur de lecture des coordonnées du point de la chaîne.\n");
                free(nouveauPoint);
                libererChaines(chaines);
                return NULL;
            }
            nouveauPoint->suiv = nouvelleChaine->points;
            nouvelleChaine->points = nouveauPoint;
        }
        nouvelleChaine->suiv = chaines->chaines;
        chaines->chaines = nouvelleChaine;
    }
    return chaines;
}

/**
 * @brief Écrit les chaînes dans un fichier.
 * @param C Chaînes à écrire.
 * @param f Pointeur vers le fichier dans lequel écrire les chaînes.
 */
void ecrireChaines(Chaines *C, FILE *f){
    fprintf(f, "NbChain: %d\nGamma: %d\n", C->nbChaines, C->gamma);
    CellChaine *chaine = C->chaines;
    while (chaine != NULL){
        fprintf(f, "%d %d\n", chaine->numero, comptePointsTotal(C));
        CellPoint *point = chaine->points;
        while (point != NULL){
            fprintf(f, "%.2f %.2f ", point->x, point->y);
            point = point->suiv;
        }
        fprintf(f, "\n");
        chaine = chaine->suiv;
    }
}

/**
 * @brief Affiche les chaînes dans un fichier SVG.
 * @param C Chaînes à afficher.
 * @param nomInstance Nom du fichier SVG.
 */
void afficheChainesSVG(Chaines *C, char* nomInstance){
    double maxx = 0,maxy = 0,minx = 1e6,miny = 1e6;
    CellChaine *ccour;
    CellPoint *pcour;
    double precx,precy;
    SVGwriter svg;
    ccour=C->chaines;
    while (ccour!=NULL){
        pcour=ccour->points;
        while (pcour!=NULL){
            if (maxx<pcour->x) maxx=pcour->x;
            if (maxy<pcour->y) maxy=pcour->y;
            if (minx>pcour->x) minx=pcour->x;
            if (miny>pcour->y) miny=pcour->y;  
            pcour=pcour->suiv;
        }
    ccour=ccour->suiv;
    }
    SVGinit(&svg,nomInstance,500,500);
    ccour=C->chaines;
    while (ccour!=NULL){
        pcour=ccour->points;
        SVGlineRandColor(&svg);
        SVGpoint(&svg,500 * (pcour->x-minx) / (maxx-minx),500 * (pcour->y-miny) / (maxy-miny)); 
        precx=pcour->x;
        precy=pcour->y;  
        pcour=pcour->suiv;
        while (pcour!=NULL){
            SVGline(&svg,500 * (precx-minx) / (maxx-minx),500 * (precy-miny) / (maxy-miny),500 * (pcour->x-minx) / (maxx-minx),500 * (pcour->y-miny) / (maxy-miny));
            SVGpoint(&svg,500 * (pcour->x-minx) / (maxx-minx),500 * (pcour->y-miny) / (maxy-miny));
            precx=pcour->x;
            precy=pcour->y;    
            pcour=pcour->suiv;
        }
        ccour=ccour->suiv;
    }
    SVGfinalize(&svg);
}

/**
 * @brief Calcule la longueur d'une chaîne.
 * @param c Chaîne dont calculer la longueur.
 * @return double Longueur de la chaîne.
 */
double longueurChaine(CellChaine *c){
    double longueur = 0.0;
    CellPoint *point1 = c->points;
    CellPoint *point2 = point1->suiv;
    while (point2 != NULL){
        longueur += distance(point1, point2);
        point1 = point2;
        point2 = point2->suiv;
    }
    return longueur;
}

/**
 * @brief Calcule la longueur totale des chaînes.
 * @param C Chaînes dont calculer la longueur totale.
 * @return double Longueur totale des chaînes.
 */
double longueurTotale(Chaines *C){
    double longueur_totale = 0.0;
    CellChaine *chaine = C->chaines;
    while (chaine != NULL){
        longueur_totale += longueurChaine(chaine);
        chaine = chaine->suiv;
    }
    return longueur_totale;
}

/**
 * @brief Compte le nombre total de points dans les chaînes.
 * @param C Chaînes dans lesquelles compter les points.
 * @return int Nombre total de points dans les chaînes.
 */
int comptePointsTotal(Chaines *C){
    int totalPoints = 0;
    CellChaine *chaine = C->chaines;
    while (chaine != NULL){
        CellPoint *point = chaine->points;
        while (point != NULL){
            totalPoints++;
            point = point->suiv;
        }
        chaine = chaine->suiv;
    }
    return totalPoints;
}

/**
 * @brief Calcule la distance entre deux points.
 * @param p1 Premier point.
 * @param p2 Deuxième point.
 * @return double Distance entre les deux points.
 */
double distance(CellPoint *p1, CellPoint *p2){
    return sqrt(pow(p2->x - p1->x, 2) + pow(p2->y - p1->y, 2));
}

/**
 * @brief Libère la mémoire allouée pour les chaînes.
 * @param C Chaînes à libérer.
 */
void libererChaines(Chaines* C){
    if (C == NULL){
        return;
    }
    CellChaine* chaine = C->chaines;
    while (chaine != NULL){
        CellChaine* chaine_suiv = chaine->suiv;
        CellPoint* point = chaine->points;
        while (point != NULL){
            CellPoint* point_suiv = point->suiv;
            free(point);
            point = point_suiv;
        }
        free(chaine);
        chaine = chaine_suiv;
    }
    free(C);
}