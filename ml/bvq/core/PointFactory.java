package ml.bvq.core;

/**
 * Factory class that creates instances of Point and LabelledPoint using arrays of features.
 *
 * @param <T>         a class that implements Point
 * @param <LabelledT> a class that extends T and implement LabelledPoint
 * @parma <Q> the type of the features contained in the points
 */
public interface PointFactory<Q, T extends Point, LabelledT extends LabeledPoint> {
    Q[] createArrayOfFeatures(int n);

    /**
     * Creates a point using an array of features.
     *
     * @param features an array of features
     * @return a point that is instance of T
     */
    T createPointFromFeatures(Q[] features);

    /**
     * Creates a matrix of point of dimension nxm.
     *
     * @param n width
     * @param m height
     * @return a matrix of points that are instances of T
     */
    T[][] createPointMatrix(int n, int m);

    /**
     * Creates a point with a label using an array of features
     *
     * @param features an array of features
     * @param label    an instance of Label
     * @return a labelled point that is instance of LabelledT
     */
    LabelledT createLabelledPointFromFeatures(Q[] features, Label label);

    LabelledT cloneWithLabel(T point, Label label);

    /**
     * Same as createPointMatrix(), but with a label
     *
     * @param n width
     * @param m height
     * @return a matrix of labelled points that are instances of LabelledT
     */
    LabelledT[][] createLabelledPointMatrix(int n, int m);

    Q getRandomValue();
}
