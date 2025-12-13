#include <stdio.h>
#include <locale.h>
#include <time.h>

int main(void)
{
  time_t maintenant = time(NULL);
  struct tm *tnow=localtime(&maintenant);
  char buff[100];
 
  strftime(buff, sizeof(buff), "%x --- %c", tnow);
  printf("Par defaut: %s\n",buff);
  
  setlocale(LC_TIME, "fr_FR.UTF-8");
  strftime(buff, sizeof(buff), "%x --- %c", tnow);
  printf("En Francais: %s\n",buff);
  return 1;
}
