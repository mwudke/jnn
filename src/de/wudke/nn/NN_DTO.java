package de.wudke.nn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NN_DTO implements Serializable {
    ArrayList<ArrayList<ArrayList<Double>>> layers;

    public NN_DTO(ArrayList<ArrayList<ArrayList<Double>>> layers) {
        this.layers = layers;
    }

    public static NN_DTO toDTO(NN nn){
        ArrayList<ArrayList<ArrayList<Double>>> layers = new ArrayList<>();

        nn.layers.forEach(l -> {
            ArrayList<ArrayList<Double>> layer = new ArrayList<>();
            l.neurons.forEach(n -> {
                ArrayList<Double> neuron = new ArrayList<>();

                neuron.add(n.bias);

                n.outputs.forEach(weight -> {
                    neuron.add(weight.weight);
                });

                layer.add(neuron);
            });
            layers.add(layer);
        });

        return new NN_DTO(layers);
    }

    public NN toNN(){
        NN nn = new NN(layers.get(0).size());

        for (int i = 1; i < layers.size(); i++){
            nn.addLayer(layers.get(i).size());
        }

        for (int i = 0; i < layers.size(); i++){
            ArrayList<ArrayList<Double>> clLayer = layers.get(i);
            Layer cLayer = nn.layers.get(i);

            for (int n = 0; n < cLayer.neurons.size(); n++){
                Neuron cNeuron = cLayer.neurons.get(n);
                ArrayList<Double> clNeuron = clLayer.get(n);

                cNeuron.bias = clNeuron.get(0);
                List<Double> outputs = clNeuron.subList(1, clNeuron.size());
                for (int m = 0; m < outputs.size(); m++){
                    cNeuron.outputs.get(m).weight = outputs.get(m);
                }
                for (int m = 0; m < outputs.size(); m++){
                    cNeuron.outputs.get(m).weight = outputs.get(m);
                }

            }
        }

        return nn;
    }
}
