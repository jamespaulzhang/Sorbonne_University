/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * La classe Agent4 représente un agent mobile dans la simulation qui peut se déplacer sur une grille
 * et interagir avec son contenu.
 */
import java.util.ArrayList;

public class Agent4 implements Mouvable {
    private int positionX;
    private int positionY;
    private ArrayList<Joyau> sacJoyaux;
    private Grille grille;

    /**
     * Constructeur de la classe Agent4.
     * @param initialX Position initiale en abscisse.
     * @param initialY Position initiale en ordonnée.
     * @param grille Grille sur laquelle l'agent évolue.
     */
    public Agent4(int initialX, int initialY, Grille grille) {
        this.positionX = initialX;
        this.positionY = initialY;
        this.grille = grille;
        this.sacJoyaux = new ArrayList<Joyau>();
    }

    /**
     * Permet à l'agent de se déplacer vers une nouvelle position spécifiée.
     * @param xnew Nouvelle position en abscisse.
     * @param ynew Nouvelle position en ordonnée.
     * @throws DeplacementIncorrectException Si le déplacement est incorrect.
     */
    public void seDeplacer(int xnew, int ynew) throws DeplacementIncorrectException {
        try {
            if (grille.sontValides(xnew, ynew)) {
                positionX = xnew;
                positionY = ynew;

                // Gérer le contenu de la case où l'agent arrive
                Contenu contenu = grille.getCase(positionX, positionY);
                if (contenu instanceof Joyau) {
                    sacJoyaux.add((Joyau) contenu);
                    grille.videCase(positionX, positionY);
                } else if (contenu instanceof Gardien) {
                    // Gérer le cas où un gardien est rencontré
                    System.out.println("Game Over: L'agent a rencontré un gardien sans force et a perdu tous ses joyaux.");
                    sacJoyaux.clear();  // Vider le sac de joyaux
                }
            } else {
                throw new DeplacementIncorrectException("Déplacement incorrect");
            }
        } catch (CoordonneesIncorrectesException | CaseNonPleineException e) {
            e.printStackTrace();  // Ou toute autre forme de gestion d'erreur
        }
    }

    /**
     * Permet à l'agent de se déplacer vers une nouvelle position spécifiée avec une force donnée.
     * @param xnew Nouvelle position en abscisse.
     * @param ynew Nouvelle position en ordonnée.
     * @param f Force de l'agent.
     * @throws DeplacementIncorrectException Si le déplacement est incorrect.
     */
    public void seDeplacer(int xnew, int ynew, int f) throws DeplacementIncorrectException {
        try {
            if (grille.sontValides(xnew, ynew)) {
                positionX = xnew;
                positionY = ynew;

                // Gérer le contenu de la case où l'agent arrive
                Contenu contenu = grille.getCase(positionX, positionY);
                if (contenu instanceof Joyau) {
                    sacJoyaux.add((Joyau) contenu);
                    grille.videCase(positionX, positionY);
                } else if (contenu instanceof Gardien) {
                    Gardien gardien = (Gardien) contenu;
                    if (gardien.getPointsDeVie() <= f) {
                        grille.videCase(positionX, positionY);
                    } else {
                        System.out.println("Game Over: L'agent a rencontré un gardien et sa force n'est pas suffisante, et a perdu tous ses joyaux.");
                        sacJoyaux.clear();  // Vider le sac de joyaux
                        gardien.setPointsDeVie(gardien.getPointsDeVie() - f);
                    }
                }
            } else {
                throw new DeplacementIncorrectException("Déplacement incorrect");
            }
        } catch (CoordonneesIncorrectesException | CaseNonPleineException e) {
            e.printStackTrace();  // Ou toute autre forme de gestion d'erreur
        }
    }

    /**
     * Calcule et renvoie la fortune totale de l'agent en additionnant les prix de tous les joyaux dans son sac.
     * @return La fortune totale de l'agent.
     */
    public int fortune() {
        int totalPrix = 0;
        for (Joyau joyau : sacJoyaux) {
            totalPrix += joyau.getPrix();
        }
        return totalPrix;
    }

    /**
     * Affiche le contenu actuel du sac de l'agent en indiquant le nombre de chaque type de joyau.
     */
    public void contenuSac() {
        int nbDiamants = 0;
        int nbOpales = 0;
        int nbRubis = 0;
    
        if (sacJoyaux.isEmpty()) {
            System.out.println("Le sac est vide.");
        } else {
            System.out.println("Contenu du sac :");
            for (Joyau joyau : sacJoyaux) {
                if (joyau != null) {
                    if (joyau.getNom() != null) {
                        switch (joyau.getNom()) {
                            case "Diamant":
                                nbDiamants++;
                                break;
                            case "Opale":
                                nbOpales++;
                                break;
                            case "Rubis":
                                nbRubis++;
                                break;
                        }
                    }
                }
            }
    
            System.out.println("Nombre de Diamants : " + nbDiamants);
            System.out.println("Nombre d'Opales : " + nbOpales);
            System.out.println("Nombre de Rubis : " + nbRubis);
        }
    }

    /**
     * Renvoie la position en abscisse de l'agent.
     * @return La position en abscisse.
     */
    public int getPositionX() {
        return this.positionX;
    }

    /**
     * Renvoie la position en ordonnée de l'agent.
     * @return La position en ordonnée.
     */
    public int getPositionY() {
        return this.positionY;
    }
}
