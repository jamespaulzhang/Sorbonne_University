#include <stdlib.h>
#include <stdio.h>

int *creer_tableau(int taille) {
  int i;
  int *p=(int *)malloc(sizeof(int)*taille);
  for (i=0; i<=taille; i++) {
    p[i]=i;
  }
  return p;
}

int main(void) {
  int i=0, t=10;
  int *p=creer_tableau(t);
  for (i=0;i<=t;i++){
    printf("p[%d]=%d, ",i,p[i]);
  }
  printf("\n");
  return 0;
}
