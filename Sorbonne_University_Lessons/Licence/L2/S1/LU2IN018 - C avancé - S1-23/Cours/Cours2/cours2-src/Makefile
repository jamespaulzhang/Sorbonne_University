progs = bug bug2 liste1_bug obscure 
flags = -Wall -g
cc= gcc

all: $(progs)

%.o: %.c
	$(cc) -c $(flags) $<

bug: bug.o
	$(cc) $(flags) -o $@ $^

bug2: bug2.o
	$(cc) $(flags) -o $@ $^

liste1_bug: liste1_bug.o
	$(cc) $(flags) -o $@ $^

obscure: obscure.o
	$(cc) $(flags) -o $@ $^

clean: 
	rm -rf *.o $(progs)
