package nn;

import java.util.Random;

/**
 * Layer, a single fully-connected layer: weights, biases, and an
 * activation function.
 *
 * @author Togzhan K.
 */
public class Layer
{
    private Matrix weights;
    private Matrix biases;
    private final Activation activation;

    private Matrix lastInput;
    private Matrix lastActivation;

    public Layer(int inputSize, int outputSize, Activation activation, Random rng)
    {
        // He-style scaling, keeps activations stable across layer sizes
        double scale = Math.sqrt(2.0 / inputSize);
        this.weights = Matrix.random(outputSize, inputSize, scale, rng);
        this.biases = Matrix.zeros(outputSize, 1);
        this.activation = activation;
    }

    public int getOutputSize()
    {
        return weights.getRows();
    }

    /** Computes weights * input + biases, then applies the activation function */
    public Matrix forward(Matrix input)
    {
        this.lastInput = input;
        Matrix z = weights.multiply(input).add(biases);
        this.lastActivation = z.map(activation::apply);
        return lastActivation;
    }

    /** dLoss/dActivation from the next layer, backpropagated through this layer */
    public Matrix backward(Matrix dLossDActivation, double learningRate)
    {
        Matrix activationDerivative = lastActivation.map(activation::derivative);
        Matrix delta = dLossDActivation.hadamard(activationDerivative);
        return applyGradients(delta, learningRate);
    }

    /** For an output layer whose loss gradient is already computed (softmax + cross-entropy) */
    public Matrix backwardWithPrecomputedDelta(Matrix delta, double learningRate)
    {
        return applyGradients(delta, learningRate);
    }

    private Matrix applyGradients(Matrix delta, double learningRate)
    {
        Matrix weightGradient = delta.multiply(lastInput.transpose());
        Matrix biasGradient = delta;
        Matrix dLossDInput = weights.transpose().multiply(delta);

        weights = weights.subtract(weightGradient.scale(learningRate));
        biases = biases.subtract(biasGradient.scale(learningRate));

        return dLossDInput;
    }
}
