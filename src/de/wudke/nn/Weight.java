package de.wudke.nn;


public class Weight{
    Neuron src, target;
    public double weight;
    public double delta;

    public Weight(Neuron src, Neuron target, double weight) {
        this.delta = 0;
        this.src = src;
        this.target = target;
        this.weight = weight;
    }

    public void commit(double lr){
        weight += delta * lr;
        delta = 0;
    }
}
