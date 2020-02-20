package ml.bvq.basic;

import ml.bvq.core.CodeVector;
import ml.bvq.core.DataGenerator;
import ml.bvq.core.Label;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;

/**
 * An extremely basic data generator that takes data from file.
 * This class is only used for testing.
 * Do not use this class for real applications!
 */
public class BasicFileDataGenerator implements DataGenerator<BasicLabeledPoint> {
    private String dataFilename;

    private ArrayList<BasicLabeledPoint> points;
    private ArrayList<BasicLabeledPoint> trainingPoints;
    private HashSet<Integer> testIndexes;
    private HashSet<Label> labels;

    private BasicPointFactory pointFactory;

    private Double[][] featuresMinMax;

    /**
     * @param dataFilename file that contains data
     */
    public BasicFileDataGenerator(BasicPointFactory pointFactory, String dataFilename) {
        this.dataFilename = dataFilename;

        this.pointFactory = pointFactory;

        points = new ArrayList<>();
        labels = new HashSet<>();
        trainingPoints = new ArrayList<>();
    }

    public void setLabels(HashSet<Label> labels) {
        this.labels = labels;
    }

    /**
     * Labels must be set before getting the training data.
     *
     * @param label label to add
     */
    public void addLabel(Label label) {
        this.labels.add(label);
    }

    /**
     * Gets a label associated with a name.
     *
     * @param label label's name
     * @return org.bvq.core.Label instance
     */
    public Label getLabel(String label) {
        assert !labels.isEmpty() : "Labels on the Data Generator are not set.";

        for (Label a : labels) {
            if (a.getLabel().equals(label.replaceAll("\"", "")))
                return a;
        }
        return null;
    }

    /**
     * Gets data from files
     *
     * @param separator   separator of the data file, if it is a csv then use its separator
     * @param labelColumn column where can be found the label
     */
    public void getDataset(String separator, int labelColumn, boolean ignoreFirstLine) {
        try (BufferedReader data = Files.newBufferedReader(Paths.get(dataFilename))) {
            String featuresString;
            boolean passedFirstLine = false;
            while ((featuresString = data.readLine()) != null) {
                if (ignoreFirstLine && !passedFirstLine) {
                    passedFirstLine = true;
                    continue;
                }

                String[] singleFeatures = featuresString.split(separator);
                Double[] features = new Double[singleFeatures.length - 1];

                // We will use this matrix for normalization of data
                if (featuresMinMax == null) {
                    featuresMinMax = new Double[singleFeatures.length - 1][2];
                    for (int i = 0; i < singleFeatures.length - 1; i++) {
                        featuresMinMax[i][0] = Double.MAX_VALUE;    // Minimum
                        featuresMinMax[i][1] = 0.0;                 // Maximum
                    }
                }

                String labelString = "";
                for (int k = 0, i = 0; i < singleFeatures.length; i++) {
                    if (i == labelColumn)
                        labelString = singleFeatures[i];
                    else {
                        features[k] = Double.parseDouble(singleFeatures[i]);

                        // Updating min and max for each features
                        if (features[k] < featuresMinMax[k][0])
                            featuresMinMax[k][0] = features[k];
                        else if (features[k] > featuresMinMax[k][1])
                            featuresMinMax[k][1] = features[k];

                        k++;
                    }
                }
                if (getLabel(labelString) != null)
                    points.add(new BasicLabeledPoint(features, getLabel(labelString)));
                else System.err.format(labelString + " not found!\n");
            }

        } catch (IOException e) {
            System.err.format("Data: IOException: %s%n", e);
        }
    }

    public void splitData(float percent, String trainingFile, String testFile) throws IOException {
        assert percent > 0.0 && percent < 1.0 : "Percent parameter must be contained in (0, 1) set.";

        testIndexes = new HashSet<>();

        long length = (int) (points.size() * percent);

        for (long i = 0; i < length; i++) {
            //org.bvq.basic.LabelledPoint p = getRandomLabelledPoint(false);
            int rand = (int) (Math.random() * (points.size()));
            if (testIndexes.contains(rand))
                i--;
            else testIndexes.add(rand);
        }

        FileWriter train = new FileWriter(trainingFile);
        FileWriter test = new FileWriter(testFile);

        for (int i = 0; i < points.size(); i++) {
            if (testIndexes.contains(i)) {
                for (double a : points.get(i).getFeatures())
                    test.write(a + ";");
                test.write(points.get(i).getLabel().getLabel() + "\n");
            } else {
                for (double a : points.get(i).getFeatures())
                    train.write(a + ";");
                train.write(points.get(i).getLabel().getLabel() + "\n");
            }
        }

        train.close();
        test.close();
    }

    /**
     * Extracts totally random code vectors from data.
     * It does not respect class distribution or anything.
     *
     * @param n number of code vectors.
     * @return code vector array.
     */
    public ArrayList<CodeVector<BasicLabeledPoint>> getRandomCodeVectors(int n) {
        ArrayList<CodeVector<BasicLabeledPoint>> cvs = new ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            BasicLabeledPoint lp = getRandomLabeledPoint(false);

            BasicLabeledPoint basicLabelledPoint =
                    pointFactory.createLabelledPointFromFeatures(lp.getFeatures(), lp.getLabel());

            CodeVector<BasicLabeledPoint> cv = new CodeVector<>(basicLabelledPoint);
            cvs.add(i, cv);
        }

        resetTrainingSet();
        return cvs;
    }

    public void normalizeData() {
        for (BasicLabeledPoint a : points)
            a.normalize(featuresMinMax);
    }

    /**
     * Get a random point from the dataset.
     *
     * @return a random labelled point.
     */
    public BasicLabeledPoint getRandomLabeledPoint(boolean withoutReplacement) {
        if (trainingPoints.size() == 0)
            resetTrainingSet();

        int rand = (int) (Math.random() * trainingPoints.size());

        BasicLabeledPoint p = trainingPoints.get(rand);

        if (withoutReplacement)
            trainingPoints.remove(p);

        return p;
    }

    /**
     * Get a random point from the dataset that has a certain class.
     * Warning: this method could cause infinite loops if there is not any point with that label.
     *
     * @return a random labelled point.
     */
    public BasicLabeledPoint getRandomLabeledPoint(boolean withoutReplacement, Label label) {
        assert labels.contains(label) : "Cannot find any label named '" + label.getLabel() + "'.";

        if (trainingPoints.size() == 0)
            resetTrainingSet();

        BasicLabeledPoint p;

        int rand;
        do {
            rand = (int) (Math.random() * trainingPoints.size());
            p = trainingPoints.get(rand);
        } while (p.getLabel() != label);

        if (withoutReplacement)
            trainingPoints.remove(p);

        return p;
    }

    @SuppressWarnings("unchecked")
    public void resetTrainingSet() {
        trainingPoints.clear();
        trainingPoints = (ArrayList<BasicLabeledPoint>) points.clone();
    }

    public ArrayList<BasicLabeledPoint> getPoints() {
        return this.points;
    }

    public int getNumberOfFeatures() {
        assert !points.isEmpty() : "There are not any point in the Data Generator.";

        return points.get(0).getFeatures().length;
    }
}
