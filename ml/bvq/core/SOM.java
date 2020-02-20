package ml.bvq.core;

import ml.bvq.core.exceptions.SOMException;

import java.util.ArrayList;

/**
 * This class does SOM clustering.
 * For the calculation of similarity it uses euclidean distance instead of scalar product.
 *
 * @param <Q> type of the values contained in the features
 * @param <P> raw data
 * @param <T> labeled data
 */
public class SOM<Q, P extends Point, T extends LabeledPoint> {
    private P[][] prototypes;
    private PointFactory<Q, P, T> pointFactory;
    private int n;

    /**
     * Initialization of the SOM with random values.
     *
     * @param numOfClusters total number of cluster. Clusters should form a squared matrix of dimension n*n.
     *                      so numOfClusters should be a number that has an integer square root.
     * @param numOfFeatures num of features of data.
     */
    public SOM(PointFactory<Q, P, T> factory, int numOfClusters, int numOfFeatures) {
        this.n = (int) Math.sqrt(numOfClusters);
        this.pointFactory = factory;
        if(n * n != numOfClusters)
            throw new SOMException("Number of clusters must have an integer square root. " +
                    "(n*n = numOfCluster where n is an int)");

        prototypes = pointFactory.createPointMatrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[] features = pointFactory.createArrayOfFeatures(numOfFeatures);
                for (int k = 0; k < numOfFeatures; k++) {
                    features[k] = pointFactory.getRandomValue();
                }
                prototypes[i][j] = pointFactory.createPointFromFeatures(features);
            }
        }
    }

    /**
     * Initialization of the SOM using DataGenerator
     *
     * @param numOfClusters total number of cluster. Clusters should form a squared matrix of dimension n*n.
     *                      so numOfClusters should be a number that has an integer square root.
     * @param dg            instance of DataGenerator interface.
     * @param label         if not null, it selects only data with a certain label
     */
    @SuppressWarnings("unchecked")
    public SOM(PointFactory<Q, P, T> factory, int numOfClusters, DataGenerator<T> dg, Label label) {
        this.pointFactory = factory;
        this.n = (int) Math.sqrt(numOfClusters);
        if(n * n != numOfClusters)
                throw new SOMException("Number of clusters must have an integer square root. " +
                        "(n*n = numOfCluster where n is an int)");

        prototypes = pointFactory.createPointMatrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                prototypes[i][j] = label != null ? (P) dg.getRandomLabeledPoint(false, label).clone():
                        (P) dg.getRandomLabeledPoint(false).clone();
            }
        }
    }

    /**
     * Get most similar prototype.
     * It uses euclidean distance for the calculation of similarity.
     *
     * @param p point to compare
     * @return an array of two elements, indicating indexes of the prototype in the matrix
     */
    public int[] getClusterOf(Point p) {
        int[] coordinates = {0, 0};
        double val_min = Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double similarityVal = p.euclideanDistance(prototypes[i][j]); //p.scalarProduct(prototypes[i][j]);

                if (similarityVal < val_min) {
                    val_min = similarityVal;
                    coordinates[0] = i;
                    coordinates[1] = j;
                }
            }
        }
        return coordinates;
    }

    /**
     * Fit values of the prototypes with the data.
     *
     * @param dg     Instance of DataGenerator interface that handles data
     * @param label  if not null, it selects only data with a certain label
     * @param t_max  maximum number of iterations
     * @param lr0    learning rate at step 0
     * @param sigma0 learning rate of the radius at step 0
     * @param alpha  constant that acts in learning rate calculation. If you do not know which is the best value, put the
     *               same value as t_max.
     * @param beta   constant that acts in radius' learning rate calculation. If you do not know which is the best value,
     *               put the same value as t_max.
     */
    @SuppressWarnings("unchecked")
    public void fit(DataGenerator<T> dg, Label label, long t_max, double lr0, double sigma0, double alpha, double beta) {
        for (int t = 0; t < t_max; t++) {
            double lr = lr0 * (t == 0 ? 1 : Math.exp(-t / alpha));
            double sigma = sigma0 * (t == 0 ? 1 : Math.exp(-t / beta));

            T p = label != null ? dg.getRandomLabeledPoint(false, label) :
                    dg.getRandomLabeledPoint(false);

            // Find the most similar neuron
            int[] coords = getClusterOf(p);

            // Update the value of all neurons
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (sigma == 0.0 && i != coords[0] && j != coords[1])
                        continue;

                    T n = (T) p.clone();
                    n.subtract(prototypes[i][j]);

                    double distance = Math.sqrt((coords[0] - i) * (coords[0] - i) + (coords[1] - j) * (coords[1] - j));
                    double radius = sigma != 0.0 ? Math.exp(-distance * distance / (2 * sigma * sigma)) : 1;
                    n.product(lr * radius);

                    prototypes[i][j].sum(n);
                }
            }
        }
    }

    public void fit(DataGenerator<T> dg, Label label, long t_max, double lr0, double sigma0) {
        fit(dg, label, t_max, lr0, sigma0, t_max, t_max);
    }

    /**
     * Converts prototypes as CodeVector.
     *
     * @param label label to associate with all prototypes.
     * @return an array of CodeVector
     */
    @SuppressWarnings("unchecked")
    public ArrayList<CodeVector<T>> getPrototypesAsCodeVectors(Label label) {
        ArrayList<CodeVector<T>> cv = new ArrayList<>(n * n + 1);
        for (int k = 0, i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                LabeledPoint lp = pointFactory.cloneWithLabel(prototypes[i][j], label);
                cv.add(k++, new CodeVector<T>((T) lp));
            }
        }
        return cv;
    }

    /**
     * Print a matrix with all the prototypes and their features.
     */
    public void printPrototypes() {
        for (int k = 0, i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(prototypes[i][j] + " ");
            }
            System.out.print("\n");
        }
    }

    /**
     * Print distribution of data in the clusters.
     *
     * @param dg instance of DataGenerator interface.
     */
    public void printDistributions(DataGenerator<T> dg) {
        if(n <= 0)
            throw new SOMException("Number of prototypes not initialized.");

        int[][] distributions = new int[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distributions[i][j] = 0;
            }
        }

        for (T p : dg.getPoints()) {
            int[] position = getClusterOf(p);
            distributions[position[0]][position[1]]++;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(distributions[i][j] + " ");
            }
            System.out.print("\n");
        }
        System.out.print("\n");
    }

    public double evaluateOverallCohesion(DataGenerator<T> dg) {
        if(n <= 0)
            throw new SOMException("Number of prototypes not initialized.");

        double cohesion = 0.0;

        for (T p : dg.getPoints()) {
            int[] position = getClusterOf(p);
            double distance = prototypes[position[0]][position[1]].euclideanDistance(p);

            cohesion += distance;
        }

        return cohesion;
    }

    public double evaluateOverallSeparation() {
        if(n <= 0)
            throw new SOMException("Number of prototypes not initialized.");

        double separation = 0.0;

        // Get first prototype
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                // Get another prototype
                for (int ii = 0; ii < n; ii++) {
                    for (int jj = 0; jj < n; jj++) {
                        if (ii == i && jj == j)
                            continue;

                        separation += prototypes[i][j].euclideanDistance(prototypes[ii][jj]);
                    }
                }
            }
        }

        return separation;
    }

    public Point[][] getPrototypes() {
        return prototypes;
    }
}
