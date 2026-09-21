#include <stdlib.h>
#include <stdio.h>

int main(void) {
  int i=0;
  int *p=i;
  *p=4;
  printf("i=%d, *p=%d\n",i,*p);
  return 0;
}
