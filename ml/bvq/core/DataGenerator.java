package ml.bvq.core;

import java.util.ArrayList;

/**
 * Interface that handles data generation.
 */
public interface DataGenerator<T extends LabeledPoint> {
    T getRandomLabeledPoint(boolean withoutReplacement);

    T getRandomLabeledPoint(boolean withoutReplacement, Label label);

    ArrayList<T> getPoints();
}