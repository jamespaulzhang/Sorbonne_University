#include <stdio.h>

void f(int n) {
  printf("n=%d\n", n);
}

int main(void) {
  void (*pf)(int); /* declaration de pf */
  pf=f; /* initialisation de pf */
  pf(3); /* affiche: n=3 */
  return 1;
}
