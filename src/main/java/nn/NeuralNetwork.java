package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * NeuralNetwork, a multilayer perceptron trained by backpropagation and
 * gradient descent, built from Matrix and Layer with no ML libraries.
 *
 * <p>The output layer either uses a plain activation with mean squared
 * error (regression-style), or softmax with cross-entropy loss
 * (classification). With softmax, the loss gradient simplifies to
 * (predicted - target), computed directly instead of through the
 * layer's usual per-neuron activation derivative.
 *
 * @author Togzhan K.
 */
public class NeuralNetwork
{
    private final List<Layer> layers;
    private final boolean softmaxOutput;

    /**
     * @param layerSizes       sizes including input and output, e.g. {784, 128, 10}
     * @param hiddenActivation activation used by all hidden layers
     * @param softmaxOutput    true for softmax + cross-entropy (classification),
     *                         false for sigmoid + mean squared error
     * @param seed             random seed for reproducible weight initialization
     */
    public NeuralNetwork(int[] layerSizes, Activation hiddenActivation, boolean softmaxOutput, long seed)
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
                // Softmax is applied afterward across the whole output vector
                activationForLayer = softmaxOutput ? Activation.LINEAR : Activation.SIGMOID;
            }
            else
            {
                activationForLayer = hiddenActivation;
            }
            layers.add(new Layer(layerSizes[i], layerSizes[i + 1], activationForLayer, rng));
        }
    }

    /** Runs a forward pass and returns the network's output for a single input */
    public Matrix predict(Matrix input)
    {
        Matrix output = input;
        for (Layer layer : layers)
        {
            output = layer.forward(output);
        }
        return softmaxOutput ? softmax(output) : output;
    }

    /** Forward pass, loss, backward pass with gradient descent updates; returns the loss */
    public double trainOnExample(Matrix input, Matrix target, double learningRate)
    {
        Matrix output = predict(input);

        double loss;
        Matrix outputDelta;
        if (softmaxOutput)
        {
            loss = crossEntropyLoss(output, target);
            outputDelta = output.subtract(target);
        }
        else
        {
            loss = meanSquaredError(output, target);
            // d(MSE)/d(output) = 2 * (output - target) / n
            outputDelta = output.subtract(target).scale(2.0 / output.getRows());
        }

        Matrix delta = outputDelta;
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            Layer layer = layers.get(i);
            boolean isOutputLayer = (i == layers.size() - 1);
            if (isOutputLayer && softmaxOutput)
            {
                delta = layer.backwardWithPrecomputedDelta(delta, learningRate);
            }
            else
            {
                delta = layer.backward(delta, learningRate);
            }
        }

        return loss;
    }

    /** Softmax over a column vector, with the standard max-subtraction for numerical stability */
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
            double p = Math.max(predicted.get(i, 0), 1e-12); // avoid log(0)
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
