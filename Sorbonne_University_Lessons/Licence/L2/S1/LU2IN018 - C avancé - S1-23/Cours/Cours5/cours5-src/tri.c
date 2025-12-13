#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*cmp_function)(const void *a, const void *b);

void triBulle(void *gtab, int nb, int taille, cmp_function cmp )
{
  int i, j;
  char *tab=(char *)gtab;
  char *tmp=(char *)malloc(taille*sizeof(char));

  for (i = nb; i > 1; i--) {
    for (j = 0; j < i - 1; j++) {
      if ((*cmp)(tab+(j + 1)*taille,tab+j*taille)<1) {
	memcpy(tmp,tab+j*taille, taille);
	memcpy(tab+j*taille, tab+(j+1)*taille, taille);
	memcpy(tab+(j+1)*taille, tmp, taille);
      }
    }
  }
  free(tmp);
}

int compare_int (const void *a, const void *b)
{
  const int *da = (const int *) a;
  const int *db = (const int *) b;
  
  return (*da > *db) - (*da < *db);
}

int compare_str(const void *a, const void *b) {
  const char **sa=(const char **)a; /* ATTENTION...*/
  const char **sb=(const char **)b; /* ATTENTION...*/
  return strcmp(*sa,*sb);
}

int main(void) {
  int t[20]={13, 3, 12, 2, 15, 9, 14, 16, 4, 5, 17, 10, 7, 8,
	     19, 1, 11, 18, 6};
  int i;
  printf("===== Tri d'entiers ====\n");
  printf("*** Avant ***\n");
  for (i=0;i<20;i++) printf("%d\n",t[i]);
  triBulle(t,20,sizeof(int),&compare_int);
  printf("*** Apres ***\n");
  for (i=0;i<20;i++) printf("%d\n",t[i]);
  
  printf("===== Tri de chaines de caracteres ====\n");
  char *ts[4]={"car", "voiture", "avion", "bateau"};
  printf("*** Avant ***\n");
  for (i=0;i<4;i++) printf("%s\n", ts[i]);
  triBulle(ts,4, sizeof(char *),&compare_str);
  printf("*** Apres ***\n");
  for (i=0;i<4;i++) printf("%s\n", ts[i]);

}
