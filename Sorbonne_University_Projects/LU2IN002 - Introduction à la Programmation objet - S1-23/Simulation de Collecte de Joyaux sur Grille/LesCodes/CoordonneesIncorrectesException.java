/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * Exception levée lorsqu'une opération est effectuée avec des coordonnées incorrectes.
 */
public class CoordonneesIncorrectesException extends Exception {
    /**
     * Constructeur de l'exception.
     * @param msg Message d'erreur associé à l'exception.
     */
    public CoordonneesIncorrectesException(String msg) {
        super(msg);
    }
}