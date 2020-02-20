package ml.bvq.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Objects;

/**
 * Class that indicates labels that could be assigned to the points.
 */
public class Label implements Serializable {
    private String label;

    // Every label has a risk map that indicates the risk of misclassificating it with another class
    private HashMap<String, Double> risks;

    /**
     * Basic constructor.
     * It assigns label name and associates the misclassification risk of itself with 0.
     *
     * @param label label name
     */
    public Label(String label) {
        this.label = label;

        risks = new HashMap<>();
        risks.put(label, 0.0);
    }

    /**
     * Associate a certain risk value with another label.
     *
     * @param misclassifiedClass the label to map
     * @param risk               its risk
     */
    public void assignMisclassificationCost(Label misclassifiedClass, double risk) {
        risks.put(misclassifiedClass.getLabel(), risk);
    }

    public double getRisk(Label misclassifiedClass) {
        return risks.getOrDefault(misclassifiedClass.getLabel(), 1.0);
    }

    public String getLabel() {
        return label;
    }

    @Override
    public boolean equals(Object b){
        if(b instanceof Label) {
            return ((Label) b).getLabel().equals(getLabel());
        }
        else
            return super.equals(b);
    }

    @Override
    public int hashCode() {
        return Objects.hash(label);
    }

    @Override
    public String toString() {
        return label;
    }
}