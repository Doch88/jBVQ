package ml.bvq.core.exceptions;

/**
 * Standard exception thrown by BVQ classes.
 */
public class BVQException extends RuntimeException {

    /**
     * Basic constructor.
     * It only assigns a message to the exception.
     * @param message error message of the exception.
     */
    public BVQException(String message) {
        super(message);
    }
}
