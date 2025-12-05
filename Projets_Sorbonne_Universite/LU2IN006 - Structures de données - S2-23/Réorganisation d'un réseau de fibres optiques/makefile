# Yuxiang ZHANG 21202829
# Antoine Lecomte 21103457

# Compilateur
CC = gcc

# Options de compilation
CFLAGS = -Wall -Wextra -std=c99

# Fichiers source
SRCS1 = ChaineMain.c Chaine.c SVGwriter.c
SRCS2 = Chaine.c Reseau.c SVGwriter.c ReconstitueReseau.c Hachage.c ArbreQuat.c Graphe.c Struct_File.c
SRCS3 = Chaine.c Reseau.c SVGwriter.c Hachage.c ArbreQuat.c temps_exec.c

# Fichiers objets correspondants
OBJS1 = $(SRCS1:.c=.o)
OBJS2 = $(SRCS2:.c=.o)
OBJS3 = $(SRCS3:.c=.o)

# Nom de l'exécutable
EXEC = ChaineMain ReconstitueReseau temps_exec

# Règle par défaut pour l'exécutable
all: $(EXEC)

# Règle de compilation pour l'exécutable
ChaineMain: $(OBJS1)
	$(CC) $^ -o $@ -lm

ReconstitueReseau: $(OBJS2)
	$(CC) $^ -o $@ -lm

temps_exec: $(OBJS3)
	$(CC) $^ -o $@ -lm

# Règle de compilation pour chaque fichier objet
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Nettoyage des fichiers objets et de l'exécutable
clean:
	rm -f $(OBJS1) $(OBJS2) $(OBJS3) $(EXEC)
