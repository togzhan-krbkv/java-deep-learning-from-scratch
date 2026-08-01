package nn;

import java.util.Random;
import java.util.function.Supplier;

/**
 * Layer, a single fully-connected layer. Accumulates gradients across a
 * batch; applyGradients() applies them through the layer's Optimizer.
 *
 * @author Togzhan K.
 */
public class Layer
{
    private Matrix weights;
    private Matrix biases;
    private final Activation activation;
    private final Optimizer weightsOptimizer;
    private final Optimizer biasesOptimizer;

    private Matrix lastInput;
    private Matrix lastActivation;

    private Matrix weightGradientSum;
    private Matrix biasGradientSum;

    public Layer(int inputSize, int outputSize, Activation activation, Random rng, Supplier<Optimizer> optimizerFactory)
    {
        double scale = Math.sqrt(2.0 / inputSize);
        this.weights = Matrix.random(outputSize, inputSize, scale, rng);
        this.biases = Matrix.zeros(outputSize, 1);
        this.activation = activation;
        this.weightsOptimizer = optimizerFactory.get();
        this.biasesOptimizer = optimizerFactory.get();
        this.weightGradientSum = Matrix.zeros(outputSize, inputSize);
        this.biasGradientSum = Matrix.zeros(outputSize, 1);
    }

    public int getOutputSize()
    {
        return weights.getRows();
    }

    public Matrix forward(Matrix input)
    {
        this.lastInput = input;
        Matrix z = weights.multiply(input).add(biases);
        this.lastActivation = z.map(activation::apply);
        return lastActivation;
    }

    public Matrix backward(Matrix dLossDActivation)
    {
        Matrix activationDerivative = lastActivation.map(activation::derivative);
        Matrix delta = dLossDActivation.hadamard(activationDerivative);
        return accumulateGradients(delta);
    }

    public Matrix backwardWithPrecomputedDelta(Matrix delta)
    {
        return accumulateGradients(delta);
    }

    private Matrix accumulateGradients(Matrix delta)
    {
        Matrix weightGradient = delta.multiply(lastInput.transpose());
        weightGradientSum = weightGradientSum.add(weightGradient);
        biasGradientSum = biasGradientSum.add(delta);
        return weights.transpose().multiply(delta);
    }

    public void applyGradients(int batchSize)
    {
        Matrix avgWeightGradient = weightGradientSum.scale(1.0 / batchSize);
        Matrix avgBiasGradient = biasGradientSum.scale(1.0 / batchSize);

        weights = weightsOptimizer.update(weights, avgWeightGradient);
        biases = biasesOptimizer.update(biases, avgBiasGradient);

        weightGradientSum = Matrix.zeros(weights.getRows(), weights.getCols());
        biasGradientSum = Matrix.zeros(biases.getRows(), biases.getCols());
    }
}
