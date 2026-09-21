//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#ifndef __ARBRE_QUAT_H__
#define __ARBRE_QUAT_H__

#include "Reseau.h"

/* Arbre quaternaire contenant les noeuds du reseau */
typedef struct arbreQuat{
    double xc, yc;          /* Coordonnees du centre de la cellule */	
    double coteX;           /* Longueur de la cellule */
    double coteY;           /* Hauteur de la cellule */
    Noeud* noeud;           /* Pointeur vers le noeud du reseau */
    struct arbreQuat *so;   /* Sous-arbre sud-ouest, pour x < xc et y < yc */
    struct arbreQuat *se;   /* Sous-arbre sud-est, pour x >= xc et y < yc */
    struct arbreQuat *no;   /* Sous-arbre nord-ouest, pour x < xc et y >= yc */
    struct arbreQuat *ne;   /* Sous-arbre nord-est, pour x >= xc et y >= yc */
} ArbreQuat;

void chaineCoordMinMax(Chaines* C, double* xmin, double* ymin, double* xmax, double* ymax);
ArbreQuat* creerArbreQuat(double xc, double yc, double coteX, double coteY);
void libererArbreQuat(ArbreQuat* a);
char trouverQuadrant(double x, double y, ArbreQuat* parent);
void insererNoeudArbre(Noeud* n, ArbreQuat** a, ArbreQuat* parent);
Noeud* rechercheNoeudParCoordonnees(Reseau* R, double x, double y);
Noeud* rechercheCreeNoeudArbre(Reseau* R, ArbreQuat** a, ArbreQuat* parent, double x, double y);
Reseau* reconstitueReseauArbre(Chaines* C);

#endif