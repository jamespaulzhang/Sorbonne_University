#include <stdio.h>
#include <stdlib.h>

void ensortant(void) {
  printf("Le programme se termine...\n");
}

int main(void) {
  atexit(ensortant);
  printf("Voila le programme...\n");
  return 1;
}
