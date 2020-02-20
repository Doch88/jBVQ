import ml.bvq.basic.BasicFileDataGenerator;
import ml.bvq.basic.BasicLabeledPoint;
import ml.bvq.basic.BasicPoint;
import ml.bvq.basic.BasicPointFactory;
import ml.bvq.core.BVQ;
import ml.bvq.core.CodeVector;
import ml.bvq.core.Label;
import ml.bvq.core.SOM;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class LocalFileTrainer {

    public static void main(String[] args) {
        // Creating Point Factory that will be used to generate instances
        BasicPointFactory pointFactory = new BasicPointFactory();

        // Initializing a org.bvq.core.DataGenerator that takes points from file
        BasicFileDataGenerator dg = new BasicFileDataGenerator(pointFactory, "examples/test.csv");
        BasicFileDataGenerator test = new BasicFileDataGenerator(pointFactory, "examples/test.csv");

        // Creating classes
        Label one = new Label("1");
        Label two = new Label("2");

        dg.addLabel(one);
        dg.addLabel(two);

        test.addLabel(one);
        test.addLabel(two);

        // Getting data from file and assigning labels to each point
        System.out.println("Getting data...");
        dg.getDataset(";", 2, true);
        test.getDataset(";", 2, true);

        dg.resetTrainingSet();

        // Initializing SOMs that will be used for the initialization of code vectors
        System.out.println("Initializing SOMs...");
        SOM<Double, BasicPoint, BasicLabeledPoint> som1 = new SOM<>(pointFactory, 25, dg, one);
        SOM<Double, BasicPoint, BasicLabeledPoint> som2 = new SOM<>(pointFactory, 25, dg, two);

        dg.resetTrainingSet();

        // Fitting SOMs to data
        som1.fit(dg, one, 200000, 0.1, 0.1);
        som2.fit(dg, two, 200000, 0.1, 0.1);

        // Unifying the sets of prototypes
        ArrayList<CodeVector<BasicLabeledPoint>> firstClass = som1.getPrototypesAsCodeVectors(one);
        ArrayList<CodeVector<BasicLabeledPoint>> secondClass = som2.getPrototypesAsCodeVectors(two);
        Set<CodeVector<BasicLabeledPoint>> set = new HashSet<>();

        set.addAll(firstClass);
        set.addAll(secondClass);

        ArrayList<CodeVector<BasicLabeledPoint>> union = new ArrayList<>(set);

        dg.resetTrainingSet();

        System.out.println("Starting BVQ...");

        // Initializing org.bvq.core.BVQ with code vectors created by SOMs
        BVQ<BasicLabeledPoint> bvq = new BVQ<>(union);

        // Let's see results of a 1-NN using only prototypes from the SOMs
        evaluate(test, bvq);

        // Start training of the org.bvq.core.BVQ algorithm
        dg.resetTrainingSet();
        bvq.fit(dg, 0.1, 2000000, 0.2);

        // Print the differences between the trained model and the raw one
        evaluate(test, bvq);
    }

    private static void evaluate(BasicFileDataGenerator dg, BVQ<BasicLabeledPoint> bvq) {
        int count = 0;
        for (int i = 0; i < dg.getPoints().size(); i++) {
            Label a = bvq.predict(dg.getPoints().get(i));
            if (a == dg.getPoints().get(i).getLabel())
                count++;
        }

        System.out.println(dg.getPoints().size() + " " + count + " " + (count * 100) / dg.getPoints().size() + "%");
    }
}