package nn;

import java.util.Random;

/**
 * Conv2DLayer, a 2D convolutional layer: a set of filters swept across
 * the input's spatial dimensions, each producing one output channel.
 * Forward pass only; backward pass is not yet implemented.
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

    private Tensor3D lastInput;
    private Tensor3D lastActivation;

    public Conv2DLayer(int inputChannels, int outputChannels, int kernelSize, int stride, int padding, Activation activation, Random rng)
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
    }

    /** Constructs a layer with explicit filters and biases, useful for testing and loading fixed weights */
    public Conv2DLayer(Tensor3D[] filters, double[] biases, int stride, int padding, Activation activation)
    {
        this.filters = filters;
        this.biases = biases;
        this.outputChannels = filters.length;
        this.inputChannels = filters[0].getChannels();
        this.kernelSize = filters[0].getHeight();
        this.stride = stride;
        this.padding = padding;
        this.activation = activation;
    }

    public Tensor3D forward(Tensor3D input)
    {
        this.lastInput = input;
        Tensor3D padded = pad(input, padding);

        int outHeight = (padded.getHeight() - kernelSize) / stride + 1;
        int outWidth = (padded.getWidth() - kernelSize) / stride + 1;
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
                                sum += filters[f].get(ic, kr, kc) * padded.get(ic, r * stride + kr, c * stride + kc);
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

    public int getOutputChannels()
    {
        return outputChannels;
    }

    public static int computeOutputSize(int inputSize, int kernelSize, int stride, int padding)
    {
        return (inputSize + 2 * padding - kernelSize) / stride + 1;
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
