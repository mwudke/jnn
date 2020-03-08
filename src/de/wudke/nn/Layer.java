package de.wudke.nn;

import java.util.ArrayList;

public class Layer{
    public ArrayList<Neuron> neurons;

    public Layer(ArrayList<Neuron> neurons) {
        this.neurons = neurons;
    }

    public double[] weights() {
        ArrayList<Double> re = new ArrayList<>();
        for (Neuron n:neurons){
            n.outputs.forEach(o -> re.add(o.weight));
        }

        double[] ret = new double[re.size()];
        for (int i = 0; i < re.size(); i++){
            ret[i] = re.get(i);
        }
        return ret;
    }

    public void commit(double lr) {
        neurons.parallelStream().forEach(n -> {
            n.commit(lr);
            n.outputs.parallelStream().forEach(o -> o.commit(lr));
        });
    }
}
