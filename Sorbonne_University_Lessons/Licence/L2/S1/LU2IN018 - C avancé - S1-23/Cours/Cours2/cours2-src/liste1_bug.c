#include <stdlib.h>
#include <stdio.h>
#include "liste1_bug.h"

/* ATTENTION ce code est volontairement bogue a des fins de
   demonstration en cours. Ne pas utiliser tel quel ! */

P_element creer_element(int data) {
  P_element nel=(P_element)malloc(sizeof(P_element));
  if (nel==NULL) {
    printf("Erreur lors de l'allocation d'un element\n");
    return NULL;
  }
  nel->data=data;
  return nel;
}

void detruire_liste(P_element liste) {
  while(liste) {
    P_element suivant = liste->suivant;
    free(liste);
    liste=suivant;
  }
}

P_element inserer_debut(P_element liste, P_element pel) {
  pel->suivant=liste;
  return pel;
}

P_element inserer_fin(P_element liste, P_element pel) {
  P_element tmp=liste;
  while(tmp->suivant!=NULL) {
    tmp=tmp->suivant;
  }
  tmp->suivant=pel;
  pel->precedent=tmp;
  return liste;
}

P_element inserer_place(P_element liste, P_element pel) {
  /* insertion en place, ordre croissant */
  P_element tmp=liste;
  if(tmp == NULL) return pel;
  if (tmp->data>pel->data) 
    return inserer_debut(liste, pel);
  while ((tmp->suivant !=NULL)&&(tmp->suivant->data<pel->data)) {
    tmp=tmp->suivant;
  }
  pel->suivant=tmp->suivant;
  pel->precedent=tmp;
  tmp->suivant->precedent=pel;
  tmp->suivant = pel;
  return liste;
}

P_element supprimer_element(P_element liste, P_element pel) {
  if (pel->precedent==NULL) {
    // suppression du debut de la liste
    P_element tmp=liste->suivant;
    tmp->precedent=NULL;
    free(liste);
    return tmp;
  }
  pel->precedent->suivant=pel->suivant;
  pel->suivant->precedent=pel->precedent;
  free(pel);
  return liste;
}

void afficher_liste(P_element liste) {
  if (liste==NULL) return;
  while(liste->precedent!=NULL) {
    liste=liste->precedent;
  }
  while(liste) {
    printf("%d ",liste->data);
  }
  printf("\n");
}

int main(void) {
  P_element l1=creer_element(5);
  l1=inserer_place(l1, creer_element(8));
  l1=inserer_place(l1, creer_element(3));
  l1=inserer_place(l1, creer_element(7));
  l1=inserer_place(l1, creer_element(1));

  afficher_liste(l1);

  return 1;
}
