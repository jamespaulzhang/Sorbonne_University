//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#ifndef __HACHAGE_H__
#define __HACHAGE_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Chaine.h"
#include "Reseau.h"

typedef struct tableHachage {
    int tailleMax;
    CellNoeud **T;
} TableHachage;

unsigned int clef(double x, double y);
unsigned int fonctionHachage(double x, double y, int tailleMax);
TableHachage* creerTableHachage(int tailleMax);
void detruireTableHachage(TableHachage* H);
void ajouterVoisin(Noeud* noeud, Noeud* voisin);
void ajouterCommodite(Reseau* reseau, Noeud* extrA, Noeud* extrB);
Noeud* rechercheCreeNoeudHachage(Reseau* R, TableHachage* H, double x, double y);
Reseau* reconstitueReseauHachage(Chaines* C, int M);

#endif