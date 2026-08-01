package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * NeuralNetwork, a sequence of Layer stages trained by backpropagation.
 * Softmax output uses cross-entropy loss; otherwise sigmoid output uses
 * mean squared error. Gradients accumulate per example; call
 * applyGradients() after a batch to update weights.
 *
 * @author Togzhan K.
 */
public class NeuralNetwork
{
    private final List<Layer> layers;
    private final boolean softmaxOutput;

    public NeuralNetwork(int[] layerSizes, Activation hiddenActivation, boolean softmaxOutput, long seed, Supplier<Optimizer> optimizerFactory)
    {
        if (layerSizes.length < 2)
        {
            throw new IllegalArgumentException("Need at least an input and an output layer");
        }

        Random rng = new Random(seed);
        this.softmaxOutput = softmaxOutput;
        this.layers = new ArrayList<>();

        for (int i = 0; i < layerSizes.length - 1; i++)
        {
            boolean isOutputLayer = (i == layerSizes.length - 2);
            Activation activationForLayer;
            if (isOutputLayer)
            {
                activationForLayer = softmaxOutput ? Activation.LINEAR : Activation.SIGMOID;
            }
            else
            {
                activationForLayer = hiddenActivation;
            }
            layers.add(new DenseLayer(layerSizes[i], layerSizes[i + 1], activationForLayer, rng, optimizerFactory));
        }
    }

    public Matrix predict(Matrix input)
    {
        Matrix output = input;
        for (Layer layer : layers)
        {
            output = layer.forward(output);
        }
        return softmaxOutput ? softmax(output) : output;
    }

    public double trainOnExample(Matrix input, Matrix target)
    {
        Matrix output = predict(input);

        double loss;
        Matrix delta;
        if (softmaxOutput)
        {
            loss = crossEntropyLoss(output, target);
            delta = output.subtract(target);
        }
        else
        {
            loss = meanSquaredError(output, target);
            delta = output.subtract(target).scale(2.0 / output.getRows());
        }

        for (int i = layers.size() - 1; i >= 0; i--)
        {
            delta = layers.get(i).backward(delta);
        }

        return loss;
    }

    public void applyGradients(int batchSize)
    {
        for (Layer layer : layers)
        {
            layer.applyGradients(batchSize);
        }
    }

    private Matrix softmax(Matrix z)
    {
        int n = z.getRows();
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++)
        {
            max = Math.max(max, z.get(i, 0));
        }

        double sumExp = 0.0;
        double[] exp = new double[n];
        for (int i = 0; i < n; i++)
        {
            exp[i] = Math.exp(z.get(i, 0) - max);
            sumExp += exp[i];
        }

        Matrix result = new Matrix(n, 1);
        for (int i = 0; i < n; i++)
        {
            result.set(i, 0, exp[i] / sumExp);
        }
        return result;
    }

    private double crossEntropyLoss(Matrix predicted, Matrix target)
    {
        double loss = 0.0;
        for (int i = 0; i < predicted.getRows(); i++)
        {
            double p = Math.max(predicted.get(i, 0), 1e-12);
            loss -= target.get(i, 0) * Math.log(p);
        }
        return loss;
    }

    private double meanSquaredError(Matrix predicted, Matrix target)
    {
        double sumSquaredError = 0.0;
        for (int i = 0; i < predicted.getRows(); i++)
        {
            double diff = predicted.get(i, 0) - target.get(i, 0);
            sumSquaredError += diff * diff;
        }
        return sumSquaredError / predicted.getRows();
    }
}
