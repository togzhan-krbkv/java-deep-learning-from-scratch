package nn;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * CnnTrainer, runs the training loop for a ConvolutionalNetwork in
 * mini-batches: shuffles each epoch, accumulates gradients over each
 * batch, then applies them.
 *
 * @author Togzhan K.
 */
public class CnnTrainer
{
    private final ConvolutionalNetwork network;
    private final int batchSize;
    private final long shuffleSeed;

    public CnnTrainer(ConvolutionalNetwork network, int batchSize, long shuffleSeed)
    {
        this.network = network;
        this.batchSize = batchSize;
        this.shuffleSeed = shuffleSeed;
    }

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
            for (int i = 0; i < trainSet.size(); i += batchSize)
            {
                int end = Math.min(i + batchSize, trainSet.size());
                for (int j = i; j < end; j++)
                {
                    MnistLoader.Example example = trainSet.get(j);
                    totalLoss += network.trainOnExample(example.image, example.target);
                }
                network.applyGradients(end - i);
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

    public double evaluate(List<MnistLoader.Example> testSet)
    {
        int correct = 0;
        for (MnistLoader.Example example : testSet)
        {
            Matrix prediction = network.predict(example.image);
            if (prediction.argmax() == example.label)
            {
                correct++;
            }
        }
        return (double) correct / testSet.size();
    }
}
