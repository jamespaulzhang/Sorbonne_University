/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * La classe Gardien représente un agent mobile de type Gardien dans la simulation.
 * Elle hérite de la classe Contenu et implémente l'interface Mouvable.
 */
public class Gardien extends Contenu implements Mouvable {
    private String nom;
    private int pointsDeVie;
    private int positionX;
    private int positionY;
    private Grille grille;

    /**
     * Constructeur de la classe Gardien.
     * @param nom Le nom du Gardien.
     * @param pointsDeVie Les points de vie du Gardien.
     * @param grille La grille sur laquelle le Gardien évolue.
     */
    public Gardien(String nom, int pointsDeVie, Grille grille) {
        super(nom, pointsDeVie);
        this.nom = nom;
        this.pointsDeVie = pointsDeVie;
        this.grille = grille;
        super.initialisePosition();
    }

    /**
     * Déplace le Gardien vers une nouvelle position spécifiée.
     * @param xnew La nouvelle coordonnée X.
     * @param ynew La nouvelle coordonnée Y.
     * @throws DeplacementIncorrectException Si le déplacement est incorrect.
     */
    public void seDeplacer(int xnew, int ynew) throws DeplacementIncorrectException {
        try {
            if (grille.sontValides(xnew, ynew)) {
                if (grille.caseEstVide(xnew, ynew)) {
                    grille.videCase(positionX, positionY);
                    positionX = xnew;
                    positionY = ynew;
                    grille.setCase(positionX, positionY, this);
                } else {
                    throw new DeplacementIncorrectException("Déplacement incorrect : la case n'est pas vide");
                }
            } else {
                throw new DeplacementIncorrectException("Déplacement incorrect : coordonnées invalides");
            }
        } catch (CoordonneesIncorrectesException | CaseNonPleineException e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Obtient le nom du Gardien.
     * @return Le nom du Gardien.
     */
    public String getNom() {
        return nom;
    }

    /**
     * Obtient les points de vie du Gardien.
     * @return Les points de vie du Gardien.
     */
    public int getPointsDeVie() {
        return pointsDeVie;
    }

    /**
     * Modifie les points de vie du Gardien.
     * @param pointsDeVie Les nouveaux points de vie.
     */
    public void setPointsDeVie(int pointsDeVie) {
        this.pointsDeVie = pointsDeVie;
    }

    /**
     * Définit la position du Gardien sur la grille.
     * @param lig La ligne de la position.
     * @param col La colonne de la position.
     */
    public void setPosition(int lig, int col) {
        this.positionX = lig;
        this.positionY = col;
    }

    /**
     * Obtient la coordonnée X de la position du Gardien.
     * @return La coordonnée X de la position du Gardien.
     */
    public int getPositionX() {
        return positionX;
    }

    /**
     * Obtient la coordonnée Y de la position du Gardien.
     * @return La coordonnée Y de la position du Gardien.
     */
    public int getPositionY() {
        return positionY;
    }
}


