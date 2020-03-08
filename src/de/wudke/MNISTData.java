package de.wudke;

public class MNISTData {
    int label;
    int[][] image;

    public MNISTData(int label, int[][] image) {
        this.label = label;
        this.image = image;
    }

    public double[] lableToV(){
        double[] v = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        v[label] = 1.0;
        return v;
    }
}
