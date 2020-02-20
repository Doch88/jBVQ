package ml.bvq.basic;

import ml.bvq.core.Label;
import ml.bvq.core.LabeledPoint;
import ml.bvq.core.Point;

/**
 * Describes a labelled point in the dataset.
 */
public class BasicLabeledPoint extends BasicPoint implements LabeledPoint {
    protected Label label;

    /**
     * Constructor that associates features with a label
     *
     * @param position features
     * @param label    class label
     */
    public BasicLabeledPoint(Double[] position, Label label) {
        super(position);
        this.label = label;
    }

    public BasicLabeledPoint(BasicPoint point, Label label) {
        super(point);
        this.label = label;
    }

    @Override
    public Label getLabel() {
        return label;
    }

    @Override
    public void setLabel(Label label) {
        this.label = label;
    }

    @Override
    public Point clone() {
        return new BasicLabeledPoint(this, label);
    }
}
