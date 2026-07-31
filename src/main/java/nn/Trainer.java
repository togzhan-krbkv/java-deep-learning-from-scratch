package nn;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Trainer, runs the training loop: shuffles each epoch, tracks average
 * loss, and reports test-set accuracy.
 *
 * @author Togzhan K.
 */
public class Trainer
{
    private final NeuralNetwork network;
    private final double learningRate;
    private final long shuffleSeed;

    public Trainer(NeuralNetwork network, double learningRate, long shuffleSeed)
    {
        this.network = network;
        this.learningRate = learningRate;
        this.shuffleSeed = shuffleSeed;
    }

    /** Trains for the given number of epochs; returns final test accuracy */
    public double train(List<MnistLoader.Example> trainSet,
                        List<MnistLoader.Example> testSet,
                        int epochs)
    {
        Random shuffleRng = new Random(shuffleSeed);
        double finalAccuracy = 0.0;

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Collections.shuffle(trainSet, shuffleRng);

            long start = System.nanoTime();
            double totalLoss = 0.0;
            for (MnistLoader.Example example : trainSet)
            {
                totalLoss += network.trainOnExample(example.input, example.target, learningRate);
            }
            double avgLoss = totalLoss / trainSet.size();
            double elapsedSec = (System.nanoTime() - start) / 1e9;

            double accuracy = evaluate(testSet);
            finalAccuracy = accuracy;

            System.out.printf(
                    "Epoch %d/%d, avg loss = %.4f, test accuracy = %.2f%%, time = %.1fs%n",
                    epoch, epochs, avgLoss, accuracy * 100.0, elapsedSec);
        }

        return finalAccuracy;
    }

    /** Fraction of test examples the network classifies correctly */
    public double evaluate(List<MnistLoader.Example> testSet)
    {
        int correct = 0;
        for (MnistLoader.Example example : testSet)
        {
            Matrix prediction = network.predict(example.input);
            if (prediction.argmax() == example.label)
            {
                correct++;
            }
        }
        return (double) correct / testSet.size();
    }
}
