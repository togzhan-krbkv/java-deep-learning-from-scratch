package nn;

import java.util.Random;
import java.util.function.Supplier;

/**
 * Conv2DLayer, a 2D convolutional layer: a set of filters swept across
 * the input's spatial dimensions, each producing one output channel.
 * Filter and bias gradients reuse Optimizer via Tensor3D's flatten and
 * unflatten, the same bridge that will later connect conv output to
 * dense layers.
 *
 * @author Togzhan K.
 */
public class Conv2DLayer
{
    private final Tensor3D[] filters;
    private final double[] biases;
    private final int inputChannels;
    private final int outputChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final Activation activation;

    private final Optimizer[] filterOptimizers;
    private final Optimizer biasOptimizer;

    private Tensor3D[] filterGradientSum;
    private double[] biasGradientSum;

    private Tensor3D lastPaddedInput;
    private Tensor3D lastActivation;

    public Conv2DLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding,
                        Activation activation, Random rng, Supplier<Optimizer> optimizerFactory)
    {
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.activation = activation;

        double scale = Math.sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        this.filters = new Tensor3D[outputChannels];
        for (int f = 0; f < outputChannels; f++)
        {
            filters[f] = randomFilter(inputChannels, kernelSize, scale, rng);
        }
        this.biases = new double[outputChannels];

        this.filterOptimizers = createOptimizers(outputChannels, optimizerFactory);
        this.biasOptimizer = optimizerFactory.get();
        resetGradientAccumulators();
    }

    /** Constructs a layer with explicit filters and biases, useful for testing and loading fixed weights */
    public Conv2DLayer(Tensor3D[] filters, double[] biases, int stride, int padding,
                        Activation activation, Supplier<Optimizer> optimizerFactory)
    {
        this.filters = filters;
        this.biases = biases;
        this.outputChannels = filters.length;
        this.inputChannels = filters[0].getChannels();
        this.kernelSize = filters[0].getHeight();
        this.stride = stride;
        this.padding = padding;
        this.activation = activation;

        this.filterOptimizers = createOptimizers(outputChannels, optimizerFactory);
        this.biasOptimizer = optimizerFactory.get();
        resetGradientAccumulators();
    }

    public Tensor3D forward(Tensor3D input)
    {
        this.lastPaddedInput = pad(input, padding);

        int outHeight = (lastPaddedInput.getHeight() - kernelSize) / stride + 1;
        int outWidth = (lastPaddedInput.getWidth() - kernelSize) / stride + 1;
        Tensor3D preActivation = new Tensor3D(outputChannels, outHeight, outWidth);

        for (int f = 0; f < outputChannels; f++)
        {
            for (int r = 0; r < outHeight; r++)
            {
                for (int c = 0; c < outWidth; c++)
                {
                    double sum = biases[f];
                    for (int ic = 0; ic < inputChannels; ic++)
                    {
                        for (int kr = 0; kr < kernelSize; kr++)
                        {
                            for (int kc = 0; kc < kernelSize; kc++)
                            {
                                sum += filters[f].get(ic, kr, kc) * lastPaddedInput.get(ic, r * stride + kr, c * stride + kc);
                            }
                        }
                    }
                    preActivation.set(f, r, c, sum);
                }
            }
        }

        this.lastActivation = preActivation.map(activation::apply);
        return lastActivation;
    }

    public Tensor3D backward(Tensor3D dLossDActivation)
    {
        Tensor3D activationDerivative = lastActivation.map(activation::derivative);
        Tensor3D delta = dLossDActivation.hadamard(activationDerivative);

        int outHeight = delta.getHeight();
        int outWidth = delta.getWidth();
        Tensor3D paddedInputGradient = Tensor3D.zeros(
                inputChannels, lastPaddedInput.getHeight(), lastPaddedInput.getWidth());

        for (int f = 0; f < outputChannels; f++)
        {
            for (int r = 0; r < outHeight; r++)
            {
                for (int c = 0; c < outWidth; c++)
                {
                    double d = delta.get(f, r, c);
                    biasGradientSum[f] += d;

                    for (int ic = 0; ic < inputChannels; ic++)
                    {
                        for (int kr = 0; kr < kernelSize; kr++)
                        {
                            for (int kc = 0; kc < kernelSize; kc++)
                            {
                                int inRow = r * stride + kr;
                                int inCol = c * stride + kc;

                                double weightGrad = d * lastPaddedInput.get(ic, inRow, inCol);
                                filterGradientSum[f].set(ic, kr, kc, filterGradientSum[f].get(ic, kr, kc) + weightGrad);

                                double inputGrad = filters[f].get(ic, kr, kc) * d;
                                paddedInputGradient.set(ic, inRow, inCol, paddedInputGradient.get(ic, inRow, inCol) + inputGrad);
                            }
                        }
                    }
                }
            }
        }

        return crop(paddedInputGradient, padding);
    }

    public void applyGradients(int batchSize)
    {
        for (int f = 0; f < outputChannels; f++)
        {
            Matrix flatFilter = filters[f].flatten();
            Matrix flatGradient = filterGradientSum[f].scale(1.0 / batchSize).flatten();
            Matrix updated = filterOptimizers[f].update(flatFilter, flatGradient);
            filters[f] = Tensor3D.unflatten(updated, inputChannels, kernelSize, kernelSize);
        }

        Matrix biasVector = Matrix.columnVector(biases);
        Matrix biasGradientVector = Matrix.columnVector(biasGradientSum).scale(1.0 / batchSize);
        Matrix updatedBias = biasOptimizer.update(biasVector, biasGradientVector);
        for (int f = 0; f < outputChannels; f++)
        {
            biases[f] = updatedBias.get(f, 0);
        }

        resetGradientAccumulators();
    }

    public int getOutputChannels()
    {
        return outputChannels;
    }

    public static int computeOutputSize(int inputSize, int kernelSize, int stride, int padding)
    {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
    }

    public double getFilterValue(int f, int c, int kr, int kc)
    {
        return filters[f].get(c, kr, kc);
    }

    public void setFilterValue(int f, int c, int kr, int kc, double value)
    {
        filters[f].set(c, kr, kc, value);
    }

    public double getBias(int f)
    {
        return biases[f];
    }

    public void setBias(int f, double value)
    {
        biases[f] = value;
    }

    public double getFilterGradient(int f, int c, int kr, int kc)
    {
        return filterGradientSum[f].get(c, kr, kc);
    }

    public double getBiasGradient(int f)
    {
        return biasGradientSum[f];
    }

    private void resetGradientAccumulators()
    {
        filterGradientSum = new Tensor3D[outputChannels];
        for (int f = 0; f < outputChannels; f++)
        {
            filterGradientSum[f] = Tensor3D.zeros(inputChannels, kernelSize, kernelSize);
        }
        biasGradientSum = new double[outputChannels];
    }

    private static Optimizer[] createOptimizers(int count, Supplier<Optimizer> factory)
    {
        Optimizer[] optimizers = new Optimizer[count];
        for (int i = 0; i < count; i++)
        {
            optimizers[i] = factory.get();
        }
        return optimizers;
    }

    private static Tensor3D pad(Tensor3D input, int padding)
    {
        if (padding == 0)
        {
            return input;
        }

        int channels = input.getChannels();
        Tensor3D padded = new Tensor3D(channels, input.getHeight() + 2 * padding, input.getWidth() + 2 * padding);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < input.getHeight(); r++)
            {
                for (int c = 0; c < input.getWidth(); c++)
                {
                    padded.set(ch, r + padding, c + padding, input.get(ch, r, c));
                }
            }
        }
        return padded;
    }

    private static Tensor3D crop(Tensor3D padded, int padding)
    {
        if (padding == 0)
        {
            return padded;
        }

        int channels = padded.getChannels();
        int height = padded.getHeight() - 2 * padding;
        int width = padded.getWidth() - 2 * padding;
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.set(ch, r, c, padded.get(ch, r + padding, c + padding));
                }
            }
        }
        return result;
    }

    private static Tensor3D randomFilter(int inputChannels, int kernelSize, double scale, Random rng)
    {
        Tensor3D filter = new Tensor3D(inputChannels, kernelSize, kernelSize);
        for (int ch = 0; ch < inputChannels; ch++)
        {
            for (int r = 0; r < kernelSize; r++)
            {
                for (int c = 0; c < kernelSize; c++)
                {
                    filter.set(ch, r, c, (rng.nextDouble() * 2 - 1) * scale);
                }
            }
        }
        return filter;
    }
}
