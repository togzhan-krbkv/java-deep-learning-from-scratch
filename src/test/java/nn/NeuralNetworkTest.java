package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkTest
{
    private static final double[][] XOR_INPUTS = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    private static final double[] XOR_TARGETS = { 0, 1, 1, 0 };

    @Test
    void learnsXorWithSgdAndBatchSizeOne()
    {
        NeuralNetwork net = new NeuralNetwork(
                new int[] { 2, 8, 1 }, Activation.SIGMOID, false, 42, () -> new SgdOptimizer(0.5));

        for (int epoch = 0; epoch < 20000; epoch++)
        {
            for (int i = 0; i < XOR_INPUTS.length; i++)
            {
                Matrix input = Matrix.columnVector(XOR_INPUTS[i]);
                Matrix target = Matrix.columnVector(new double[] { XOR_TARGETS[i] });
                net.trainOnExample(input, target);
                net.applyGradients(1);
            }
        }

        assertXorConverged(net);
    }

    @Test
    void learnsXorWithFullBatchGradientDescent()
    {
        NeuralNetwork net = new NeuralNetwork(
                new int[] { 2, 8, 1 }, Activation.SIGMOID, false, 42, () -> new SgdOptimizer(0.5));

        for (int epoch = 0; epoch < 20000; epoch++)
        {
            for (int i = 0; i < XOR_INPUTS.length; i++)
            {
                Matrix input = Matrix.columnVector(XOR_INPUTS[i]);
                Matrix target = Matrix.columnVector(new double[] { XOR_TARGETS[i] });
                net.trainOnExample(input, target);
            }
            net.applyGradients(XOR_INPUTS.length);
        }

        assertXorConverged(net);
    }

    @Test
    void learnsXorWithAdam()
    {
        NeuralNetwork net = new NeuralNetwork(
                new int[] { 2, 8, 1 }, Activation.SIGMOID, false, 42, () -> new AdamOptimizer(0.05));

        for (int epoch = 0; epoch < 5000; epoch++)
        {
            for (int i = 0; i < XOR_INPUTS.length; i++)
            {
                Matrix input = Matrix.columnVector(XOR_INPUTS[i]);
                Matrix target = Matrix.columnVector(new double[] { XOR_TARGETS[i] });
                net.trainOnExample(input, target);
            }
            net.applyGradients(XOR_INPUTS.length);
        }

        assertXorConverged(net);
    }

    @Test
    void softmaxOutputSumsToOne()
    {
        NeuralNetwork net = new NeuralNetwork(
                new int[] { 4, 6, 3 }, Activation.RELU, true, 7, () -> new SgdOptimizer(0.1));
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
        double[][] inputs = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        double[][] targets = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

        NeuralNetwork net = new NeuralNetwork(
                new int[] { 3, 5, 3 }, Activation.RELU, true, 11, () -> new SgdOptimizer(0.1));

        double firstLoss = 0.0;
        for (int i = 0; i < inputs.length; i++)
        {
            firstLoss += net.trainOnExample(Matrix.columnVector(inputs[i]), Matrix.columnVector(targets[i]));
            net.applyGradients(1);
        }

        double lastLoss = firstLoss;
        for (int epoch = 0; epoch < 200; epoch++)
        {
            lastLoss = 0.0;
            for (int i = 0; i < inputs.length; i++)
            {
                lastLoss += net.trainOnExample(Matrix.columnVector(inputs[i]), Matrix.columnVector(targets[i]));
                net.applyGradients(1);
            }
        }

        assertTrue(lastLoss < firstLoss, "first=" + firstLoss + ", last=" + lastLoss);
    }

    private void assertXorConverged(NeuralNetwork net)
    {
        for (int i = 0; i < XOR_INPUTS.length; i++)
        {
            Matrix input = Matrix.columnVector(XOR_INPUTS[i]);
            double predicted = net.predict(input).get(0, 0);
            assertEquals(XOR_TARGETS[i], predicted, 0.1);
        }
    }
}
