#ifndef ENTREESORTIELC_H
#define ENTREESORTIELC_H

#include "biblioLC.h"

Biblio* charger_n_entrees(char* nomfic, int n);
void enregistrer_biblio(Biblio* b, char* nomfic);
void afficher_livre(Livre* livre);
void afficher_biblio(Biblio* b);
Livre* rechercher_par_numero(Biblio* b, int n);
Livre* rechercher_par_titre(Biblio* b, char* titre);
Biblio* rechercher_par_auteur(Biblio* b, char* auteur);
void supprimer_livre(Biblio* biblio, int num, char* titre, char* auteur);
void fusionner_bibliotheques(Biblio* biblio1, Biblio* biblio2);
Biblio* rechercher_exemplaires_multiples(Biblio* biblio);

#endif