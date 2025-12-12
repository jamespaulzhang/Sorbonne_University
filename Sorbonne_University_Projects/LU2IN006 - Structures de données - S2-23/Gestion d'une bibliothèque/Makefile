all: main

main: main.c biblioLC.o entreeSortieLC.o biblioH.o entreeSortieH.o
	gcc -Wall -o main main.c biblioLC.o biblioH.o entreeSortieLC.o entreeSortieH.o

biblioLC.o: biblioLC.c
	gcc -Wall -c biblioLC.c

biblioH.o: biblioH.c
	gcc -Wall -c biblioH.c

entreeSortieLC.o: entreeSortieLC.c
	gcc -Wall -c entreeSortieLC.c

entreeSortieH.o: entreeSortieH.c
	gcc -Wall -c entreeSortieH.c

clean:
	rm -f main *.o

