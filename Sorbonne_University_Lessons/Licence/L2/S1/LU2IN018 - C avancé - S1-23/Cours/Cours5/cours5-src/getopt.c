#include <unistd.h>
#include <stdio.h>

int main (int argc, char ** argv) {
  int c;
  while ((c=getopt(argc,argv,"abcd:"))!=-1) {
    if(c=='?')
      printf("Option inconnue !\n");
    else {
      printf("Option %c\n",c);
      if (c=='d') 
	printf("  argument: %s\n",optarg);   
    }
  }
}
