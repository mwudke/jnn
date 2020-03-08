package de.wudke.nn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class Util {

    public static double sigmoid(double x){
        return 1/(1 + Math.pow(Math.E, (x* -1)));
    }

    public static double sigmoidPrime(double x){
        double n = sigmoid(x);
        return n * (1-n);
    }

    public static double[] costDerivative(double[] output_activations, double[] y){
        double [] cost = new double[y.length];

        for (int i = 0; i < y.length; i++){
            cost[i] = y[i] - output_activations[i];
        }

        return cost;
    }

    public static void exportNN(String filename, NN nn){
        FileOutputStream fos;
        ObjectOutputStream out;
        try {
            fos = new FileOutputStream(filename);
            out = new ObjectOutputStream(fos);
            out.writeObject(NN_DTO.toDTO(nn));

            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static NN importNN(String filename){
        FileInputStream fis;
        ObjectInputStream in;
        NN_DTO nn = null;
        try {
            fis = new FileInputStream(filename);
            in = new ObjectInputStream(fis);
            nn = (NN_DTO) in.readObject();
            in.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        return nn.toNN();
        }
}
