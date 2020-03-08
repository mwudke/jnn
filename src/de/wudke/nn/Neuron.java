package de.wudke.nn;

import java.util.ArrayList;

public class Neuron{
    public double bias;
    public double value;
    public double z;
    public double delta;
    public ArrayList<Weight> inputs, outputs;

    public Neuron() {
        this.bias = 0;
        this.z = 0;
        this.value = 0;
        this.delta = 0;
        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();
    }

    public void evaluate(){
        double sum = 0;

        for (Weight w: inputs){
            sum += (w.src.value * w.weight);
        }

        z = sum + bias;
        value = Util.sigmoid(sum + bias);
    }

    public void evaluate(double init){
       value = Util.sigmoid(init + bias);
       z = init + bias;
    }

    public void commit(double lr){
        bias += delta * lr;
        delta = 0;
    }
}
