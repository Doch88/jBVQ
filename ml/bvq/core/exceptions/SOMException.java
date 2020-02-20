package ml.bvq.core.exceptions;

/**
 * Standard exception thrown by SOM clustering.
 */
public class SOMException extends RuntimeException {

    /**
     * Basic constructor.
     * It only assigns a message to the exception.
     * @param message error message of the exception.
     */
    public SOMException(String message) {
        super(message);
    }
}
