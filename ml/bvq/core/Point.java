package ml.bvq.core;

import java.io.Serializable;

/**
 * A generic point containing a set of features.
 */
public interface Point extends Serializable, Cloneable {
    /**
     * Sum two points and returns the result.
     *
     * @param secondPoint second operand of the operation
     * @return result
     */
    Point sum(Point secondPoint);

    /**
     * Subtract two points and returns the result.
     *
     * @param secondPoint second operand of the operation
     * @return result
     */
    Point subtract(Point secondPoint);

    /**
     * Multiplies a point with a scalar and returns the result.
     *
     * @param scalar a scalar to multiply.
     * @return result
     */
    Point product(double scalar);

    /**
     * Gets the half-way point between two points.
     *
     * @param secondPoint second point of the operations.
     * @return half-way point
     */
    Point getMiddlePoint(Point secondPoint);

    /**
     * Does a scalar product between two points.
     *
     * @param secondPoint second operand of the operation
     * @return scalar product
     */
    double scalarProduct(Point secondPoint);

    /**
     * Gets the euclidean distance between two points.
     *
     * @param secondPoint second operand of the operation
     * @return euclidean distance
     */
    double euclideanDistance(Point secondPoint);

    int getNumOfFeatures();

    /**
     * Gets the norm of a point.
     *
     * @return norm
     */
    double norm();

    /**
     * Clones this point.
     *
     * @return a point having same features of this point.
     */
    Point clone();
}
