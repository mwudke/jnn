package de.wudke;


import de.wudke.nn.*;
import mnist.MnistReader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Main {
    //Pfande zu MNIST Daten und Model Ordner
    private static final String MNIST_DATA = "C:\\workspace\\jnn\\src\\resources\\mnist\\";
    private static final String MODEL_DATA = "C:\\workspace\\jnn\\src\\resources\\model\\";

    private static int[] trainLabels, testLabels;
    private static List<int[][]> trainImages, testImages;
    private static ArrayList<MNISTData> mnistTrainData, mnistTestData;

    private static NN nn;

    public static void main(String[] args) {

        //Neues netz erstellen
        nn = new NN(784);
        nn.addLayer(30).addLayer(10);


        // ODER bestehendes Model laden
//        nn = Util.importNN(MODEL_DATA + "mnist_bigboy.nn");

        //laden der MNIST Daten
        loadMnistData();


        //ausführen einer Prediction auf ein zufällig gewählten MNIST Satz mit ASCII ausgabe des Bildes
//        testPredict(nn);

        //Auswertung der Trefferquote des Netzes auf den MNIST Testdaten
        testNN(nn);

        //Netz trainieren nach gradient decent: epochAnzahl, Batchsize, Lernrate, Auswertung nach jeden epoch?
        sgd(28, 10, 0.35, true);


        //exportieren des Models
        Util.exportNN(MODEL_DATA + "mnist_nn12.nn", nn);
    }

    public static void testPredict(NN nn) {
        Random r = new Random();
        int imageIndex = r.nextInt(testImages.size());

        System.out.println(MnistReader.renderImage(testImages.get(imageIndex)));
        System.out.println(testLabels[imageIndex]);


        int[] array = Stream.of(testImages.get(imageIndex)).flatMapToInt(IntStream::of).toArray();
        Double[] dArray = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            dArray[i] = (double) array[i];
        }

        double[] prediction = nn.predict(dArray);

        int bestGuessIndex = 0;
        double lastBestGuess = 0;
        for (int i = 0; i < prediction.length; i++) {
            System.out.println(i + ": " + prediction[i]);

            if (prediction[i] > lastBestGuess) {
                bestGuessIndex = i;
                lastBestGuess = prediction[i];
            }
        }

        System.out.println("Prediction: " + bestGuessIndex);
    }

    private static void loadMnistData() {
        //Lade Mnist Daten
        trainLabels = MnistReader.getLabels(MNIST_DATA + "train-labels.idx1-ubyte");
        trainImages = MnistReader.getImages(MNIST_DATA + "train-images.idx3-ubyte");
        mnistTrainData = new ArrayList<>();
        for (int i = 0; i < trainLabels.length; i++) {
            mnistTrainData.add(new MNISTData(trainLabels[i], trainImages.get(i)));
        }

        testLabels = MnistReader.getLabels(MNIST_DATA + "t10k-labels.idx1-ubyte");
        testImages = MnistReader.getImages(MNIST_DATA + "t10k-images.idx3-ubyte");
        mnistTestData = new ArrayList<>();
        for (int i = 0; i < testLabels.length; i++) {
            mnistTestData.add(new MNISTData(testLabels[i], testImages.get(i)));
        }
    }

    private static void testNN(NN nn) {
        int correct = 0;
        for (int i = 0; i < testImages.size(); i++) {

            int[] array = Stream.of(testImages.get(i)).flatMapToInt(IntStream::of).toArray();
            Double[] dArray = new Double[array.length];
            for (int n = 0; n < array.length; n++) {
                dArray[n] = Double.valueOf(array[n]);
            }

            double[] prediction = nn.predict(dArray);

            int bestGuessIndex = 0;
            double lastBestGuess = 0;
            for (int n = 0; n < prediction.length; n++) {

                if (prediction[n] > lastBestGuess) {
                    bestGuessIndex = n;
                    lastBestGuess = prediction[n];
                }
            }

            if (bestGuessIndex == testLabels[i]) {
                correct++;
            }

        }

        Double per = 100.0 / (double) testImages.size() * (double) correct;
        per = (double) Math.round(per * 1000) / 1000;
        System.out.println(correct + "/" + testImages.size() + ": \t" + per + "%");
    }

    private static void sgd(int epochs, int miniBatchSize, double lr, boolean testEpoch) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            ArrayList<MNISTData> mnistT = (ArrayList<MNISTData>) mnistTrainData.clone();
            Collections.shuffle(mnistT);

            ArrayList<ArrayList<MNISTData>> miniBatches = new ArrayList<>();
            for (int i = 0; i < mnistT.size() / miniBatchSize; i++) {
                miniBatches.add(new ArrayList<>(mnistT.subList(i, i + miniBatchSize)));
            }

            miniBatches.forEach(mb -> updateMiniBatch(mb, lr));

            if (testEpoch) {
                System.out.print("Epoch: " + epoch + " abgeschlossen: \t");
                testNN(nn);
            } else {
                System.out.println("Epoch: " + epoch + " abgeschlossen");
            }
        }
    }

    private static void updateMiniBatch(ArrayList<MNISTData> mb, double lr) {
        mb.forEach(Main::backprop);
        nn.commit(lr / (double) mb.size());
    }

    private static void backprop(MNISTData md) {
        //vorwärts
        int[] array = Stream.of(md.image).flatMapToInt(IntStream::of).toArray();
        Double[] dArray = new Double[array.length];
        for (int n = 0; n < array.length; n++) {
            dArray[n] = Double.valueOf(array[n]);
        }

        nn.predict(dArray);


        //rückwärts

        //init delta
        Layer outputLayer = nn.layers.get(nn.layers.size() - 1);
        double[] deltas = new double[outputLayer.neurons.size()];
        double[] soll = md.lableToV();
        for (int i = 0; i < deltas.length; i++) {
            //delta w = f'(z) * (soll - ist)
            deltas[i] = Util.sigmoidPrime(outputLayer.neurons.get(i).z) * (soll[i] - outputLayer.neurons.get(i).value);
            outputLayer.neurons.get(i).delta += deltas[i];
        }

        //output delta weights
        Layer preOutputLayer = nn.layers.get(nn.layers.size() - 2);
        for (int i = 0; i < preOutputLayer.neurons.size(); i++) {
            for (int j = 0; j < deltas.length; j++) {
                preOutputLayer.neurons.get(i).outputs.get(j).delta += deltas[j] * preOutputLayer.neurons.get(i).value;
            }
        }

        for (int l = nn.layers.size() - 2; l > 0; l--) {
            Layer currentLayer = nn.layers.get(l);
            Layer preLayer = nn.layers.get(l - 1);

            //Update deltas
            double[] nDeltas = new double[currentLayer.neurons.size()];
            for (int i = 0; i < currentLayer.neurons.size(); i++) {
                double sum = 0;
                for (int n = 0; n < deltas.length; n++){
                    sum += deltas[n] * currentLayer.neurons.get(i).outputs.get(n).weight;
                }
                nDeltas[i] = Util.sigmoidPrime(currentLayer.neurons.get(i).z) * sum;
                currentLayer.neurons.get(i).delta += nDeltas[i];
            }
            deltas = nDeltas;

            //delta weights
            for (int i = 0; i < preLayer.neurons.size(); i++) {
                for (int j = 0; j < deltas.length; j++) {
                    preLayer.neurons.get(i).outputs.get(j).delta += deltas[j] * preLayer.neurons.get(i).value;
                }
            }
        }
    }
}
