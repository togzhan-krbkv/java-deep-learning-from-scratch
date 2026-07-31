package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest
{
    @Test
    void learnsXorWithSigmoidAndMeanSquaredError()
    {
        // XOR is not linearly separable, so this proves backprop works
        double[][] inputs = {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 0 };

        NeuralNetwork net = new NeuralNetwork(new int[] { 2, 8, 1 }, Activation.SIGMOID, false, 42);

        for (int epoch = 0; epoch < 20000; epoch++)
        {
            for (int i = 0; i < inputs.length; i++)
            {
                Matrix input = Matrix.columnVector(inputs[i]);
                Matrix target = Matrix.columnVector(new double[] { targets[i] });
                net.trainOnExample(input, target, 0.5);
            }
        }

        for (int i = 0; i < inputs.length; i++)
        {
            Matrix input = Matrix.columnVector(inputs[i]);
            double predicted = net.predict(input).get(0, 0);
            double expected = targets[i];
            assertEquals(expected, predicted, 0.1,
                    "XOR(" + (int) inputs[i][0] + ", " + (int) inputs[i][1] + ") should be about " + expected);
        }
    }

    @Test
    void softmaxOutputSumsToOne()
    {
        NeuralNetwork net = new NeuralNetwork(new int[] { 4, 6, 3 }, Activation.RELU, true, 7);
        Matrix input = Matrix.columnVector(new double[] { 0.5, -0.2, 0.9, 0.1 });

        Matrix output = net.predict(input);

        double sum = 0.0;
        for (int i = 0; i < output.getRows(); i++)
        {
            assertTrue(output.get(i, 0) >= 0.0 && output.get(i, 0) <= 1.0);
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void crossEntropyLossDecreasesWithTraining()
    {
        // Trivial 3-class one-hot problem: loss should drop with training
        double[][] inputs = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        double[][] targets = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

        NeuralNetwork net = new NeuralNetwork(new int[] { 3, 5, 3 }, Activation.RELU, true, 11);

        double firstLoss = 0.0;
        for (int i = 0; i < inputs.length; i++)
        {
            firstLoss += net.trainOnExample(Matrix.columnVector(inputs[i]), Matrix.columnVector(targets[i]), 0.1);
        }

        double lastLoss = firstLoss;
        for (int epoch = 0; epoch < 200; epoch++)
        {
            lastLoss = 0.0;
            for (int i = 0; i < inputs.length; i++)
            {
                lastLoss += net.trainOnExample(Matrix.columnVector(inputs[i]), Matrix.columnVector(targets[i]), 0.1);
            }
        }

        assertTrue(lastLoss < firstLoss, "Loss should decrease with training (first=" + firstLoss + ", last=" + lastLoss + ")");
    }
}
