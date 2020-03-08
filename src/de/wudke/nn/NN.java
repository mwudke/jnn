package de.wudke.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NN {
    public ArrayList<Layer> layers = new ArrayList();

    public NN(int inputLayerSize) {
        ArrayList<Neuron> inputLayerNeurons = new ArrayList<>();
        for (int i = 0; i < inputLayerSize; i++){
            inputLayerNeurons.add(new Neuron());
        }

        Layer inputLayer = new Layer(inputLayerNeurons);
        layers.add(inputLayer);
    }

    public NN addLayer(int size){
        ArrayList<Neuron> neurons = new ArrayList<>();
        Layer preLayer = layers.get(layers.size()-1);

        for (int i = 0; i < size; i++){
            Neuron n = new Neuron();
            neurons.add(n);

            Random ran = new Random();

            n.bias = ran.nextDouble() * 2 -1;

            for (Neuron pn: preLayer.neurons){
                Weight w = new Weight(pn, n, ran.nextDouble() * 2 -1);  //init weight
                pn.outputs.add(w);
                n.inputs.add(w);
            }
        }

        layers.add(new Layer(neurons));

        return this;
    }

    public double[] predict(Double[] inputs){
        ArrayList<Neuron> inputLayer = layers.get(0).neurons;

        for (int i = 0; i < inputLayer.size(); i++){
            inputLayer.get(i).evaluate(inputs[i]);
        }

        List<Layer> evalLayers = layers.subList(1,layers.size());
        for (Layer l: evalLayers){
            l.neurons.parallelStream().forEach(n -> n.evaluate());
        }

        Layer outputLayer = layers.get(layers.size()-1);
        double[] valueList = new double[outputLayer.neurons.size()];
        for (int i = 0; i < outputLayer.neurons.size(); i++){
            valueList[i] = outputLayer.neurons.get(i).value;
        }

        return valueList;
    }

    public ArrayList<Neuron> getAllNeurons(){
        ArrayList<Neuron> nl = new ArrayList<>();
        for (Layer l: layers){
            nl.addAll(l.neurons);
        }
        return nl;
    }

    public ArrayList<Neuron> getAllHNeurons(){
        ArrayList<Neuron> nl = new ArrayList<>();
        for (int i = 1; i < layers.size(); i++){
            nl.addAll(layers.get(i).neurons);
        }
        return nl;
    }

    public ArrayList<Weight> getAllWeights(){
        ArrayList<Weight> wl = new ArrayList<>();
        for (Layer l: layers){
            l.neurons.forEach(n -> wl.addAll(n.outputs));
        }
        return wl;
    }

    public void commit(double lr){
        layers.parallelStream().forEach(layer -> layer.commit(lr));
    }

}
