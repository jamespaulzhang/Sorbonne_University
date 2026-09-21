#ifndef ENTREESORTIEH_H
#define ENTREESORTIEH_H

#include "biblioH.h"

BiblioH* charger_n_entreesH(char* nomfic, int n);
void enregistrer_biblioH(BiblioH* b, char* nomfic);
void afficher_livreH(LivreH* livre);
void afficher_biblioH(BiblioH* b);
LivreH* rechercher_par_numeroH(BiblioH* b, int num);
LivreH* rechercher_par_titreH(BiblioH* b, char* titre);
BiblioH* rechercher_par_auteurH(BiblioH* b, char* auteur);
void supprimer_livreH(BiblioH* biblio, int num, char* titre, char* auteur);
void fusionner_bibliothequesH(BiblioH* biblio1, BiblioH* biblio2);
BiblioH* rechercher_exemplaires_multiplesH(BiblioH* biblio);

#endif /* ENTREESORTIEH_H */