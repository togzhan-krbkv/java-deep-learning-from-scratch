package nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * MainCnn, trains a small convolutional network on MNIST:
 * Conv(1,8,3x3) -> Pool(2x2) -> Conv(8,16,3x3) -> Pool(2x2) -> Flatten ->
 * Dense(784,64) -> Dense(64,10) -> Softmax.
 *
 * Expects data/mnist_train.csv and data/mnist_test.csv in Kaggle's
 * CSV format (label,pixel0,...,pixel783), with a header row.
 *
 * @author Togzhan K.
 */
public class MainCnn
{
    public static void main(String[] args) throws IOException
    {
        String trainPath = "data/mnist_train.csv";
        String testPath = "data/mnist_test.csv";
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

        Random rng = new Random(seed);
        Supplier<Optimizer> optimizerFactory = () -> new AdamOptimizer(learningRate);

        List<Tensor3DLayer> convStack = new ArrayList<>();
        convStack.add(new Conv2DLayer(1, 8, 3, 1, 1, Activation.RELU, rng, optimizerFactory));
        convStack.add(new MaxPoolLayer(2, 2));
        convStack.add(new Conv2DLayer(8, 16, 3, 1, 1, Activation.RELU, rng, optimizerFactory));
        convStack.add(new MaxPoolLayer(2, 2));

        int flattenedSize = 16 * 7 * 7;
        List<Layer> denseStack = new ArrayList<>();
        denseStack.add(new DenseLayer(flattenedSize, 64, Activation.RELU, rng, optimizerFactory));
        denseStack.add(new DenseLayer(64, MnistLoader.NUM_CLASSES, Activation.LINEAR, rng, optimizerFactory));

        ConvolutionalNetwork network = new ConvolutionalNetwork(convStack, denseStack);
        CnnTrainer trainer = new CnnTrainer(network, batchSize, seed);

        System.out.printf("Training for %d epochs, batch size %d, learning rate %.4f%n%n", epochs, batchSize, learningRate);

        double finalAccuracy = trainer.train(trainSet, testSet, epochs);

        System.out.printf("%nFinal test accuracy: %.2f%%%n", finalAccuracy * 100.0);
    }
}
