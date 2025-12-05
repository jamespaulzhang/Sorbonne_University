/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * Exception levée lorsqu'une tentative de déplacement est incorrecte.
 */
public class DeplacementIncorrectException extends Exception {
    /**
     * Constructeur de l'exception.
     * @param msg Message d'erreur associé à l'exception.
     */
    public DeplacementIncorrectException(String msg) {
        super(msg);
    }
}