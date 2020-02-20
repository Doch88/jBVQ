package ml.bvq.core;

import java.util.*;

/**
 * Class that contains all the methods necessary to build a confusion matrix and to compute its metrics.
 * It also contains a method that can be used to check the variation of the accuracy in time.
 */
public class ConfusionMatrix {
    private Map<Label, Map<Label, Integer>> confusionMatrix;
    private List<Integer> lastClassifications;
    private List<Double> lastF1ScoreOfClass;

    private Label monitoredClass = null;

    private double maxAccuracy = 0.55;
    private double maxF1ScoreOfClass = 0.55;
    private double lastAccuracy = 0.0;
    private double lastF1Score = 0.0;

    private int window = 500;

    private int iterations = 0;

    private int minIterations = 0;

    /**
     * Constructor that allows to check the variation of accuracy in time.
     * @param minIterations minIterations to wait before calculating the variation of accuracy. If too low
     *                      it can be affected by the fluctuations of the first period.
     */
    public ConfusionMatrix(int minIterations) {
        confusionMatrix = Collections.synchronizedMap(new HashMap<>());
        lastClassifications = Collections.synchronizedList(new ArrayList<>());
        lastF1ScoreOfClass = Collections.synchronizedList(new ArrayList<>());
        this.minIterations = minIterations;
    }

    /**
     * Basic constructor.
     */
    public ConfusionMatrix() {
        confusionMatrix = Collections.synchronizedMap(new HashMap<>());
        lastClassifications = Collections.synchronizedList(new ArrayList<>());
        lastF1ScoreOfClass = Collections.synchronizedList(new ArrayList<>());
    }

    /**
     * Add a classification result to the confusion matrix.
     * @param trueLabel the groundtruth label of the testing point
     * @param classification label that the classifier assigned to the point
     */
    public void addTestingResult(Label trueLabel, Label classification) {
        confusionMatrix.putIfAbsent(trueLabel, Collections.synchronizedMap(new HashMap<>()));
        confusionMatrix.get(trueLabel).putIfAbsent(classification, 0);
        int metric = confusionMatrix.get(trueLabel).get(classification) + 1;
        confusionMatrix.get(trueLabel).put(classification, metric);

        if(classification == trueLabel)
            lastClassifications.add(1);
        else
            lastClassifications.add(0);

        if(lastClassifications.size() > window)
            lastClassifications.remove(0);
    }

    /**
     * Get all the values associated with a groundtruth class.
     * @param label a groundtruth label
     * @return number of occurrences of the specified class
     */
    public int getTotalOfLabel(Label label) {
        if(!confusionMatrix.containsKey(label))
            return 0;

        int counter = 0;

        for (Integer i: confusionMatrix.get(label).values()) {
            counter += i;
        }

        return counter;
    }

    /**
     * Get all the positives values associated with a predicted label.
     * @param label a predicted label
     * @return number of positive predictions that have been classified with a certain label
     */
    public int getPositivesOfLabel(Label label) {
        int counter = 0;

        for (Map<Label, Integer> results: confusionMatrix.values()) {
            counter += results.getOrDefault(label, 0);
        }

        return counter;
    }

    /**
     * Get all the negatives values associated with a predicted label.
     * @param label a predicted label
     * @return number of negative predictions that have been classified with a certain label
     */
    public int getNegativesOfLabel(Label label) {
        int counter = 0;

        for (Label trueLabel: confusionMatrix.keySet()) {
            if(!trueLabel.equals(label)) {
                for(Integer i: confusionMatrix.get(trueLabel).values()) {
                    counter += i;
                }
            }
        }

        return counter;
    }

    /**
     * Count predictions added to the confusion matrix
     * @return number of predictions
     */
    public int countPredictions() {
        int counter = 0;
        for(Map<Label, Integer> map : confusionMatrix.values()) {
            for(Integer i : map.values()) {
                counter += i;
            }
        }
        return counter;
    }

    /**
     * Gets the number of predictions correctly made of a certain label
     * @param label a label to check
     * @return the number of correctly predicted elements
     */
    public int getTruePositivesOfLabel(Label label) {
        if(!confusionMatrix.containsKey(label))
            return 0;

        return confusionMatrix.get(label).getOrDefault(label, 0);
    }

    /**
     * Gets the overall accuracy of the confusion matrix.
     * It is calculated using the diagonal of the matrix, divided by the
     * total number of predictions (see countPredictions).
     * @return a double value between 0 and 1
     */
    public double getAccuracy() {
        if(countPredictions() == 0)
            return 0;

        int counter = 0;
        for(Label trueLabel : confusionMatrix.keySet()) {
            counter += confusionMatrix.get(trueLabel).getOrDefault(trueLabel, 0);
        }

        double accuracy = ((double)counter)/ ((double)countPredictions());

        lastAccuracy = calcAccuracyOnLastClassifications();

        if(iterations < minIterations)
            iterations++;

        if(calcAccuracyOnLastClassifications() > maxAccuracy && iterations >= minIterations)
            maxAccuracy = calcAccuracyOnLastClassifications();

        return accuracy;
    }

    /**
     * Gets the recall associated with a certain label.
     * Recall is a metric calculated using true positives and total number of real predictions.
     * @param label a class to analyze
     * @return a double value between 0 and 1
     */
    public double getRecall(Label label) {
        if(getTotalOfLabel(label) <= 0)
            return 0.0;
        return ((double)getTruePositivesOfLabel(label))/((double)getTotalOfLabel(label));
    }

    /**
     * Gets the precision associated with a certain label.
     * Precision is a metric calculated using true positives and total number of positives of a class.
     * @param label a class to analyze
     * @return a double value between 0 and 1
     */
    public double getPrecision(Label label) {
        if(getPositivesOfLabel(label) <= 0)
            return 0.0;
        return ((double)getTruePositivesOfLabel(label))/((double)getPositivesOfLabel(label));
    }

    /**
     * Gets the variation between the last accuracy value and the max accuracy occurred.
     * @return difference between the max accuracy and the last one.
     */
    public double getVariationOfAccuracy() {
        if(maxAccuracy - lastAccuracy > 0 && iterations >= minIterations) {
            return maxAccuracy - lastAccuracy;
        }

        return 0.0;
    }

    public double getVariationOfF1Score() {
        if(maxF1ScoreOfClass - lastF1Score > 0 && iterations >= minIterations) {
            return maxF1ScoreOfClass - lastF1Score;
        }

        return 0.0;
    }

    public double calcAccuracyOnLastClassifications(){
        if(lastClassifications == null)
            lastClassifications = Collections.synchronizedList(new ArrayList<>());
        int count = 0;
        for (Integer i : lastClassifications)
            if (i != null)
                count += i;
        return ((double)count)/lastClassifications.size();
    }

    public double calcF1ScoreOnLastClassifications(){
        if(lastF1ScoreOfClass == null)
            lastF1ScoreOfClass = Collections.synchronizedList(new ArrayList<>());
        int count = 0;
        double sum = 0.0;
        for (Double i : lastF1ScoreOfClass)
            if (i != null) {
                count += 1;
                sum += i;
            }
        return sum/((double)count);
    }

    /**
     * Reset variation of accuracy
     */
    public void resetVariation() {
        maxAccuracy = 0.55;
        maxF1ScoreOfClass = 0.55;
        iterations = 0;
    }

    public void resetMatrix() {
        confusionMatrix = Collections.synchronizedMap(new HashMap<>());
    }

    public void setMonitoredClass(Label monitoredClass) {
        this.monitoredClass = monitoredClass;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    /**
     * Metrics that mixes recall and precision in an unique value.
     * @param label class to analyze
     * @return a value between 0 and 1
     */
    public double getF1Score(Label label) {
        if((getRecall(label) + getPrecision(label)) <= 0)
            return 0.0;
        double score = 2 * (((getRecall(label)) * getPrecision(label))/(getRecall(label) + getPrecision(label)));
        if(monitoredClass != null && label == monitoredClass) {
            lastF1ScoreOfClass.add(score);
            if(lastF1ScoreOfClass.size() > window)
                lastF1ScoreOfClass.remove(0);
            double average = calcF1ScoreOnLastClassifications();

            lastF1Score = average;

            if(average > maxF1ScoreOfClass && iterations >= minIterations)
                maxF1ScoreOfClass = average;
        }
        return score;
    }
}
