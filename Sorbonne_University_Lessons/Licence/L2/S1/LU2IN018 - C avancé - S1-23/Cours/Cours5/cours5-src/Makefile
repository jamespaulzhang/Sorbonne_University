GCC_FLAGS=-Wall -g
GCC_LIB_FLAGS=-fpic -static
CC=gcc
EXE=atexit getopt locale map mini_printf pointeur_fonction tri

all: $(EXE)

%.o: %.c
	$(CC) $(GCC_FLAGS) -c $^


atexit: atexit.o
	$(CC) $(GCC_FLAGS) -o $@ $^

getopt: getopt.o
	$(CC) $(GCC_FLAGS) -o $@ $^

locale: locale.o
	$(CC) $(GCC_FLAGS) -o $@ $^

map: map.o
	$(CC) $(GCC_FLAGS) -o $@ $^

mini_printf: mini_printf.o
	$(CC) $(GCC_FLAGS) -o $@ $^

pointeur_fonction: pointeur_fonction.o
	$(CC) $(GCC_FLAGS) -o $@ $^


tri: tri.o
	$(CC) $(GCC_FLAGS) -o $@ $^




clean:
	rm -f *.o $(EXE) $(LIB) *~ 
