#include<stdio.h>
#include<stdarg.h>

void ecritentier(int n) {

  int m=n%10;

  if (n>=10) {
    ecritentier(n/10);
  }
  putchar(m+'0');

}


void mini_printf(char *fmt, ...) {

  va_list pa;
  char *p,*vals;
  int vali;  

  va_start(pa, fmt);

  for( p=fmt ; *p!='\0' ; p++) {
    if (*p != '%') {putchar(*p);continue;
    }
    p=p+1;
    switch (*p) {
    case 'd' : vali = va_arg(pa, int);
      ecritentier(vali);
      break;
    case 's' : 
      for( vals = va_arg(pa, char *); *vals!='\0'; vals++)
	putchar(*vals);
      break;
    default  : putchar(*p);
      break;
    }
  }
  
  va_end(pa);
}

int main() {
  mini_printf("debut_format entier %d puis chaine %s\n", 
	      3, "la_chaine");
  return 0;
}
