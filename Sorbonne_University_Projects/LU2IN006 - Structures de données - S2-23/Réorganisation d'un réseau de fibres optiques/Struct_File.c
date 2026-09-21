//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdlib.h>
#include <stdio.h>
#include "Struct_File.h"

/**
 * @brief Initialise une file.
 * @param f File à initialiser.
 */
void Init_file(S_file *f){
  f->tete=NULL;
  f->dernier=NULL;
}

/**
 * @brief Vérifie si la file est vide.
 * @param f File à vérifier.
 * @return int Retourne 1 si la file est vide, sinon 0.
 */
int estFileVide(S_file *f){
  return f->tete == NULL;
}

/**
 * @brief Enfile un élément dans la file.
 * @param f File dans laquelle enfiler.
 * @param donnee Élément à enfiler.
 */
void enfile(S_file * f, int donnee){
 Cellule_file *nouv=(Cellule_file *) malloc(sizeof(Cellule_file));
  nouv->val=donnee;
  nouv->suiv=NULL;
  if (f->tete==NULL)
    f->tete=nouv;
  else
    f->dernier->suiv=nouv;
  f->dernier=nouv;
}

/**
 * @brief Défile un élément de la file.
 * @param f File dans laquelle défiler.
 * @return int Élément défilé.
 */
int defile(S_file *f){
  int v=f->tete->val;
  Cellule_file *temp=f->tete;
  if (f->tete==f->dernier)
    f->dernier=NULL;
  f->tete=f->tete->suiv;
  free(temp);
  return v;
}
