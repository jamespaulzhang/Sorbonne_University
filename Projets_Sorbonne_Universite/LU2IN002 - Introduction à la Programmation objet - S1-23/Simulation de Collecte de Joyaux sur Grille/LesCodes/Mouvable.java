/**
 * Yuxiang ZHANG, numero etudiant: 21202829
 * Antoine LECOMTE, numero etudiant: 21103457
 * Groupe 4
 */

/** 
 * L'interface Mouvable définit le comportement des objets pouvant se déplacer dans la simulation.
 */
public interface Mouvable {
    /**
     * Déplace l'objet vers une nouvelle position spécifiée.
     * 
     * @param xnew La nouvelle position en abscisse.
     * @param ynew La nouvelle position en ordonnée.
     * @throws DeplacementIncorrectException Si le déplacement est incorrect.
     */
    public void seDeplacer(int xnew, int ynew) throws DeplacementIncorrectException;
}