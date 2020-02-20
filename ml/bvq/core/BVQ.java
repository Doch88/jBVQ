package ml.bvq.core;

import ml.bvq.core.exceptions.BVQException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Main class of the algorithm.
 * It handles the array of the code vectors and has the methods for the training, the prediction and the evaluation.
 */
public class BVQ<T extends LabeledPoint> implements Serializable {
    /**
     * An interface created to simply define a learning rate lambda function
     */
    protected interface LR extends Serializable {
        double getLR(double lr0, int i);
    }

    /**
     * Default learning rate function as defined by the original paper
     */
    protected LR learningRateFunction = (lr0, i) -> lr0 * (i == 0 ? 1 : Math.pow(i, -0.51));

    /**
     * Array containing current code vectors
     */
    protected List<CodeVector<T>> codeVectors;

    protected CodeVector<T> lastUpdatedCodeVector;

    /**
     * Map that indicates number of code vector needed for each class.
     * If all the values of the map are not 0 then the classifier can not be trained
     */
    protected HashMap<Label, Integer> numOfCodeVectors = null;

    private int last_i = 0;

    /**
     * Basic constructor.
     *
     * @param codeVectors initial code vectors.
     */
    public BVQ(List<CodeVector<T>> codeVectors) {
        this.codeVectors = codeVectors;
    }

    /**
     * Constructor used when the code vector are lazy instantiated
     *
     * @param numOfCodeVectors number of code vectors needed for each class.
     */
    @SuppressWarnings("unchecked")
    public BVQ(HashMap<Label, Integer> numOfCodeVectors) {
        this.numOfCodeVectors = (HashMap<Label, Integer>) numOfCodeVectors.clone();
        this.codeVectors = new CopyOnWriteArrayList<>();
    }

    /**
     * Get the two nearest code vectors to a point.
     *
     * @param point the point to be searched.
     * @return an array of the two code vectors.
     */
    protected List<CodeVector<T>> getNearestCodeVectors(T point) {
        if(codeVectors.size() <= 1)
            throw new BVQException("There aren't enough code vectors. There must be at least 2 code vectors.");
        if(point == null)
            throw new BVQException("Null point passed to BVQ.");

        // Initializing ArrayList with two null elements
        List<CodeVector<T>> nearest = new ArrayList<>(2);
        nearest.add(0, null);
        nearest.add(1, null);

        // Set initial distances to the maximum value in order to minimize distance
        double[] distances = {Double.MAX_VALUE, Double.MAX_VALUE};

        // Searching for the nearest code vectors
        for (CodeVector<T> a : codeVectors) {
            if(a == null)
                throw new BVQException("Null code vector in the code vector list.");
            if(a.getPoint() == null)
                throw new BVQException("Code vector containing null point.");

            double distance = a.getPoint().euclideanDistance(point);

            if (distance < distances[0]) {
                distances[1] = distances[0];
                nearest.set(1, nearest.get(0));
                distances[0] = distance;
                nearest.set(0, a);
            } else if (distance < distances[1]) {
                nearest.set(1, a);
                distances[1] = distance;
            }
        }
        return nearest;
    }

    /**
     * Start the training of the algorithm.
     *
     * @param data  an instance of the interface org.bvq.core.DataGenerator. It returns the training points
     * @param lr0   learning rate at the iteration 0.
     * @param n_max maximum number of iterations.
     * @param delta threshold of the distance between a point and its projection on the classification surface.
     */
    public void fit(DataGenerator<T> data, double lr0, long n_max, double delta) {
        if(codeVectors.size() <= 1)
            throw new BVQException("There aren't enough code vectors. There must be at least 2 code vectors.");

        for (int i = 0; i < n_max; i++) {
            T trainingPoint = data.getRandomLabeledPoint(false);

            train(trainingPoint, lr0, i, delta);
        }
    }

    /**
     * Do a single training step.
     *
     * @param trainingPoint the point used to train.
     * @param lr0           learning rate at iteration 0
     * @param i             current number of iteration
     * @param delta         threshold of the distance between a point and its projection on the classification surface.
     */
    public boolean train(T trainingPoint, double lr0, int i, double delta) {
        if(codeVectors.size() <= 1)
            throw new BVQException("There aren't enough code vectors. There must be at least 2 code vectors.");
        if(!isReady())
            throw new BVQException("Code vectors are not fully instantiated.");
        if(trainingPoint == null)
            throw new BVQException("Training point equal to null.");

        double learningRate = learningRateFunction.getLR(lr0, i);

        List<CodeVector<T>> nearest = getNearestCodeVectors(trainingPoint);

        if(nearest.get(0) == null || nearest.get(1) == null)
                throw new BVQException("Something weird happened, nearest code vectors are equal to null.");

        lastUpdatedCodeVector = nearest.get(0);

        last_i = i;

        // Let's update the value of the two nearest code vectors
        return nearest.get(0).update(nearest.get(1), trainingPoint, learningRate, delta);
    }

    /**
     * A method that allow to define a new learning rate function.
     *
     * @param learningRateFunction a lambda function that takes lr0 and i as parameters
     */
    public void setLearningRateFunction(LR learningRateFunction) {
        this.learningRateFunction = learningRateFunction;
    }

    /**
     * Predict a point class.
     *
     * @param point class to be predicted.
     * @return label of the class
     */
    public Label predict(T point) {
        if(!isReady())
            throw new BVQException("Code vectors are not initialized.");
        if(point == null)
            throw new BVQException("Trying to predict a null point.");

        List<CodeVector<T>> nearest = getNearestCodeVectors(point);
        return nearest.get(0).getPoint().getLabel();
    }

    /**
     * Add a code vector only if there aren't enough code vectors for its class.
     *
     * @param cv a code vector to add
     * @return true if the code vector has been added, false otherwise
     */
    public boolean addCodeVector(CodeVector<T> cv) {
        if(numOfCodeVectors == null)
            throw new BVQException("Number of code vectors are not specified, cannot add code vector.");
        if(cv == null || cv.getPoint() == null)
            return false;

        Label label = cv.getPoint().getLabel();

        if(numOfCodeVectors.get(label) == null)
            throw new BVQException("Label not initialized.");

        if(numOfCodeVectors.get(label) > 0) {
            codeVectors.add(cv);
            numOfCodeVectors.replace(label, numOfCodeVectors.get(label)-1);
            return true;
        } else
            return false;
    }

    /**
     * The classifier is ready when all the code vectors are instantiated.
     *
     * @return true if there are enough code vectors (or it was used the base constructor), false otherwise
     */
    public boolean isReady(){
        if(numOfCodeVectors == null)
            return true;

        for(Integer i : numOfCodeVectors.values())
            if(i > 0) return false;
        return true;
    }

    @SuppressWarnings("unchecked")
    public void resetCodeVectors(HashMap<Label, Integer> numOfCodeVectors) {
        this.numOfCodeVectors = (HashMap<Label, Integer>) numOfCodeVectors.clone();
        this.codeVectors = new CopyOnWriteArrayList<>();
    }

    public CodeVector<T> getLastUpdatedCodeVector() {
        return lastUpdatedCodeVector;
    }

    public int getLastI() {
        return last_i;
    }

    /**
     * Print all the code vector and their features.
     */
    public void printCodeVectors() {
        for (CodeVector cv : codeVectors) {
            System.out.println(cv);
        }
    }

    public List<CodeVector<T>> getCodeVectors() {
        return codeVectors;
    }

}