set terminal pngcairo enhanced font 'Verdana,12'
set output 'plot.png'

set title "Temps de recherche en fonction de la taille de la bibliothèque"
set xlabel "Taille de la bibliothèque"
set ylabel "Temps de recherche (secondes)"

plot "data.txt" using 1:2 with linespoints title "Liste chaînée", \
     "data.txt" using 1:3 with linespoints title "Tableau de hachage"
