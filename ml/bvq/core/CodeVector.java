package ml.bvq.core;

import ml.bvq.core.exceptions.BVQException;

import java.io.Serializable;

/**
 * Class that contains a Labelled Point. This class describes a CodeVector.
 * The set of CodeVector creates Voronoi regions. Every Voronoi region classifies the points that it contains.
 * This class uses "Container" design pattern.
 */
public class CodeVector<T extends LabeledPoint> implements Serializable {
    private T point;

    /**
     * Basic constructor.
     */
    public CodeVector(T point) {
        this.point = point;
    }

    /**
     * Creates a classification surface between two code vectors and project a point on it.
     *
     * @param secondCodeVector the other code vector.
     * @param point            a point to project
     * @return the point projected on the classification surface
     */
    private Point projectPoint(CodeVector<T> secondCodeVector, Point point) {
        Point projected = point.clone();

        // The vector normal to the decision surface
        Point n = this.getPoint().clone();
        n.subtract(secondCodeVector.getPoint());

        // Taking the middle point between the two code vectors
        Point middlePoint = this.getPoint().getMiddlePoint(secondCodeVector.getPoint());

        // Subtracting the middle point to the point to project
        middlePoint.subtract(point);

        // Calculating the scalar product between last result and the vector normal
        double numerator = middlePoint.scalarProduct(n);

        // The scalar product of a vector with itself is its squared magnitude.
        double denominator = n.scalarProduct(n);

        n.product(numerator / denominator);
        projected.sum(n);

        return projected;
    }

    /**
     * This method updates the position (features) of the code vector.
     * It must be used with the nearest code vector to a point. The code vector will be updated only if the distance
     * between the point and its projection is lower than delta/2.
     * This method uses Stochastic Gradient Descent.
     *
     * @param secondCodeVector second nearest code vector to the point
     * @param trainingPoint    the point that updates position of the code vectors.
     * @param lr               learning rate at this iteration. It must be updated outside this method.
     * @param delta            threshold of distance between the training point and its projection
     * @return true if update succeeded, false otherwise.
     */
    public boolean update(CodeVector<T> secondCodeVector, LabeledPoint trainingPoint, double lr, double delta) {
        Point projection = projectPoint(secondCodeVector, trainingPoint);

        // Calculating the distance between the point and its projection
        Point condition = trainingPoint.clone();
        condition.subtract(projection);

        // The distance should be lesser than delta/2
        if (condition.norm() <= delta / 2) {
            Point n = this.getPoint().clone();

            // Stochastic Gradient Descent
            double norm = n.subtract(secondCodeVector.getPoint()).norm();
            double beta = (trainingPoint.getLabel().getRisk(secondCodeVector.getPoint().getLabel()) -
                    trainingPoint.getLabel().getRisk(this.getPoint().getLabel())) / (delta * norm);
            beta *= lr;

            // Updating first nearest code vector
            Point niValue = this.getPoint().clone();
            niValue.subtract(projection);
            niValue.product(beta);
            this.getPoint().subtract(niValue);

            // Updating second nearest code vector
            Point njValue = secondCodeVector.getPoint().clone();
            njValue.subtract(projection);
            njValue.product(beta);
            secondCodeVector.getPoint().sum(njValue);
            return true;
        }
        return false;
    }

    public T getPoint() {
        return point;
    }

    @Override
    public String toString() {
        return "CodeVector{" +
                "features=[" + point +
                "], class=" + this.getPoint().getLabel().getLabel() + '}';
    }
}