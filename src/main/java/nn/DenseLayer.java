package nn;

import java.util.Random;
import java.util.function.Supplier;

/**
 * DenseLayer, a fully-connected Layer: weights, biases, and an
 * activation function.
 *
 * @author Togzhan K.
 */
public class DenseLayer implements Layer
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

    public DenseLayer(int inputSize, int outputSize, Activation activation, Random rng, Supplier<Optimizer> optimizerFactory)
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

    @Override
    public Matrix forward(Matrix input)
    {
        this.lastInput = input;
        Matrix z = weights.multiply(input).add(biases);
        this.lastActivation = z.map(activation::apply);
        return lastActivation;
    }

    @Override
    public Matrix backward(Matrix dLossDActivation)
    {
        Matrix activationDerivative = lastActivation.map(activation::derivative);
        Matrix delta = dLossDActivation.hadamard(activationDerivative);

        Matrix weightGradient = delta.multiply(lastInput.transpose());
        weightGradientSum = weightGradientSum.add(weightGradient);
        biasGradientSum = biasGradientSum.add(delta);

        return weights.transpose().multiply(delta);
    }

    @Override
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
