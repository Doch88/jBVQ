package ml.bvq.basic;

import ml.bvq.core.Label;
import ml.bvq.core.PointFactory;

public class BasicPointFactory implements PointFactory<Double, BasicPoint, BasicLabeledPoint> {

    @Override
    public BasicPoint createPointFromFeatures(Double[] features) {
        return new BasicPoint(features);
    }

    @Override
    public Double[] createArrayOfFeatures(int n) {
        return new Double[n];
    }

    @Override
    public BasicPoint[][] createPointMatrix(int n, int m) {
        return new BasicPoint[n][m];
    }

    @Override
    public BasicLabeledPoint createLabelledPointFromFeatures(Double[] features, Label label) {
        return new BasicLabeledPoint(features, label);
    }

    @Override
    public BasicLabeledPoint[][] createLabelledPointMatrix(int n, int m) {
        return new BasicLabeledPoint[n][m];
    }

    @Override
    public BasicLabeledPoint cloneWithLabel(BasicPoint point, Label label) {
        return new BasicLabeledPoint(point, label);
    }

    @Override
    public Double getRandomValue() {
        return Math.random();
    }
}
