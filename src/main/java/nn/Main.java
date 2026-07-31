package nn;

import java.io.IOException;
import java.util.List;

/**
 * Main, trains a 784, 128, 10 network on MNIST.
 *
 * Expects data/mnist_train.csv and data/mnist_test.csv in Kaggle's
 * CSV format (label,pixel0,...,pixel783), with a header row.
 *
 * @author Togzhan K.
 */
public class Main
{
    public static void main(String[] args) throws IOException
    {
        String trainPath = "data/mnist_train.csv";
        String testPath = "data/mnist_test.csv";
        int hiddenSize = 128;
        int epochs = 10;
        int batchSize = 32;
        double learningRate = 0.001;
        long seed = 42;

        System.out.println("MNIST train data from " + trainPath);
        List<MnistLoader.Example> trainSet = MnistLoader.load(trainPath, true);
        System.out.println("  Loaded " + trainSet.size());

        System.out.println("MNIST test data from " + testPath);
        List<MnistLoader.Example> testSet = MnistLoader.load(testPath, true);
        System.out.println("  Loaded " + testSet.size());

        int[] architecture = {
                MnistLoader.IMAGE_SIZE, hiddenSize, MnistLoader.NUM_CLASSES
        };
        System.out.printf(
                "Building network: architecture = %s, hidden activation = ReLU, output = softmax + cross-entropy, optimizer = Adam%n",
                java.util.Arrays.toString(architecture));

        NeuralNetwork network = new NeuralNetwork(
                architecture, Activation.RELU, true, seed, () -> new AdamOptimizer(learningRate));
        Trainer trainer = new Trainer(network, batchSize, seed);

        System.out.printf(
                "Training for %d epochs, batch size %d, learning rate %.4f%n%n", epochs, batchSize, learningRate);

        double finalAccuracy = trainer.train(trainSet, testSet, epochs);

        System.out.printf("%nFinal test accuracy: %.2f%%%n", finalAccuracy * 100.0);
    }
}
