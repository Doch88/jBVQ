package ml.bvq.core;

/**
 * This class represents a point that has a class assigned.
 */
public interface LabeledPoint extends Point {

    /**
     * Returns the label associated with this point.
     *
     * @return an instance of Label
     */
    Label getLabel();

    /**
     * Set the label of the point
     *
     * @param label an instance of Label
     */
    void setLabel(Label label);
}
