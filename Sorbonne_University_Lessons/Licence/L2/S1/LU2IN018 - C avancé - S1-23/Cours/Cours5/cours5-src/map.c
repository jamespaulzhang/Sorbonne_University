#include <stdio.h>

typedef void (*fonctionSurEntier)(int);

void map (fonctionSurEntier f, int *tableau, int taille) {
  unsigned int i;
  for (i=0;i<taille;i++) {
    f(tableau[i]);  
  }
}

void print_int(int i) {
  printf("Element : %d\n",i);
}

int main(void) {
  int tab[10]={0,1,2,3,4,5,6,7,8,9};
  map(print_int,tab,10);
}
