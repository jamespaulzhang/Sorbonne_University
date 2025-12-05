/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * Exception levée lorsqu'une opération qui nécessite une case non pleine est effectuée sur une case pleine.
 */
public class CaseNonPleineException extends Exception {
    /**
     * Constructeur de l'exception.
     * @param msg Message d'erreur associé à l'exception.
     */
    public CaseNonPleineException(String msg) {
        super(msg);
    }
}
