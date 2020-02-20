package ml.bvq.basic;

import ml.bvq.core.Point;

import java.util.Arrays;

/**
 * Describes a point in the dataset.
 */
public class BasicPoint implements Point {
    protected Double[] features;

    public BasicPoint(Double[] features) {
        this.features = features;
    }

    /**
     * Clone another point.
     *
     * @param p point to clone.
     */
    public BasicPoint(BasicPoint p) {
        this.features = new Double[p.getFeatures().length];
        for (int i = 0; i < features.length; i++) {
            features[i] = p.getFeatures()[i];
        }
    }

    /**
     * Do the scalar product with another point.
     * It asserts that the two points have the same size.
     *
     * @param secondPoint second operand of the operation
     * @return the scalar product
     */
    @Override
    public double scalarProduct(Point secondPoint) {
        assert secondPoint.getNumOfFeatures() == features.length :
                "Operands of scalar product must have same dimensions.";

        double scalarProd = 0.0;
        for (int i = 0; i < features.length; i++) {
            scalarProd += features[i] * convert(secondPoint).getFeature(i);
        }
        return scalarProd;
    }

    /**
     * Element-wise sum between two points.
     * Results will be saved on this instance and returned by the method.
     *
     * @param secondPoint second operand of the operation.
     * @return itself
     */
    @Override
    public Point sum(Point secondPoint) {
        assert secondPoint.getNumOfFeatures() == features.length :
                "Operands of element-wise sum must have same dimensions.";

        for (int i = 0; i < features.length; i++) {
            features[i] += convert(secondPoint).getFeature(i);
        }
        return this;
    }

    /**
     * Element-wise subtraction between two points.
     * Results will be saved on this instance and returned by the method.
     *
     * @param secondPoint second operand of the operation.
     * @return itself
     */
    @Override
    public Point subtract(Point secondPoint) {
        assert secondPoint.getNumOfFeatures() == features.length :
                "Operands of element-wise subtraction must have same dimensions.";

        for (int i = 0; i < features.length; i++) {
            features[i] -= convert(secondPoint).getFeature(i);
        }
        return this;
    }

    /**
     * Multiply its features with a scalar.
     * Results will be saved on this instance and returned by the method.
     *
     * @param scalar scalar of the operation.
     * @return itself
     */
    @Override
    public Point product(double scalar) {
        for (int i = 0; i < features.length; i++) {
            features[i] *= scalar;
        }
        return this;
    }

    /**
     * Get the halfway point between two points.
     *
     * @param secondPoint the second operand of the operation.
     * @return a new point that contains features of the halfway point.
     */
    @Override
    public Point getMiddlePoint(Point secondPoint) {
        BasicPoint n = (BasicPoint) this.clone();
        n.sum(secondPoint);
        n.product(0.5);
        return n;
    }

    /**
     * Euclidean distance between two points.
     *
     * @param secondPoint the second operand of the operation.
     * @return a double that indicates the distance.
     */
    @Override
    public double euclideanDistance(Point secondPoint) {
        assert secondPoint.getNumOfFeatures() == features.length :
                "Operands of euclidean distance must have same dimensions.";

        double distance = 0.0;
        for (int i = 0; i < features.length; i++) {
            distance += (features[i] - convert(secondPoint).getFeature(i)) * (features[i] - convert(secondPoint).getFeature(i));
        }
        return Math.sqrt(distance);

    }

    /**
     * Normalize a point using a matrix containing min and max values for each feature
     *
     * @param featuresMinMax the matrix containing the values
     */
    public void normalize(Double[][] featuresMinMax) {
        assert featuresMinMax.length == features.length && featuresMinMax.length > 0 :
                "Matrix containing min and max values of the features must have same dimension of the feature vector.";
        assert featuresMinMax[0].length == 2 :
                "Each element of the min-max matrix must have exactly 2 elements (min and max)";

        for (int i = 0; i < features.length; i++) {
            features[i] = (features[i] - featuresMinMax[i][0]) /
                    (featuresMinMax[i][1] - featuresMinMax[i][0]);
        }
    }

    /**
     * @return norm of the features.
     */
    @Override
    public double norm() {
        double value = 0.0;
        for (double feature : features) {
            value += feature * feature;
        }
        return Math.sqrt(value);
    }

    private static BasicPoint convert(Point point) {
        assert point instanceof BasicPoint : "Second operand must be of the same type of first operand.";

        return (BasicPoint) point;
    }

    public Double[] getFeatures() {
        return features;
    }

    public Double getFeature(int i) {
        return features[i];
    }

    @Override
    public int getNumOfFeatures() {
        return features.length;
    }

    /**
     * Clone this point.
     */
    @Override
    public Point clone() {
        return new BasicPoint(this);
    }

    @Override
    public String toString() {
        return "org.bvq.basic.Point{" +
                "features=" + Arrays.toString(features) +
                '}';
    }
}