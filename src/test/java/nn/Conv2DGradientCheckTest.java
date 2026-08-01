package nn;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class Conv2DGradientCheckTest
{
    private static final double EPSILON = 1e-5;
    private static final double TOLERANCE = 1e-4;

    @Test
    void filterGradientsMatchNumericalGradients()
    {
        Random rng = new Random(3);
        Conv2DLayer layer = new Conv2DLayer(2, 3, 3, 1, 1, Activation.SIGMOID, rng, () -> new SgdOptimizer(0.01));
        Tensor3D input = randomTensor(2, 6, 6, rng);
        Tensor3D upstreamGradient = randomTensor(3, 6, 6, rng);

        layer.forward(input);
        layer.backward(upstreamGradient);

        for (int f = 0; f < 3; f++)
        {
            for (int c = 0; c < 2; c++)
            {
                for (int kr = 0; kr < 3; kr++)
                {
                    for (int kc = 0; kc < 3; kc++)
                    {
                        double analytical = layer.getFilterGradient(f, c, kr, kc);
                        double numerical = numericalFilterGradient(layer, input, upstreamGradient, f, c, kr, kc);
                        assertEquals(numerical, analytical, TOLERANCE,
                                "f=" + f + " c=" + c + " kr=" + kr + " kc=" + kc);
                    }
                }
            }
        }
    }

    @Test
    void biasGradientsMatchNumericalGradients()
    {
        Random rng = new Random(5);
        Conv2DLayer layer = new Conv2DLayer(2, 3, 3, 1, 1, Activation.SIGMOID, rng, () -> new SgdOptimizer(0.01));
        Tensor3D input = randomTensor(2, 6, 6, rng);
        Tensor3D upstreamGradient = randomTensor(3, 6, 6, rng);

        layer.forward(input);
        layer.backward(upstreamGradient);

        for (int f = 0; f < 3; f++)
        {
            double analytical = layer.getBiasGradient(f);
            double numerical = numericalBiasGradient(layer, input, upstreamGradient, f);
            assertEquals(numerical, analytical, TOLERANCE, "f=" + f);
        }
    }

    @Test
    void inputGradientsMatchNumericalGradients()
    {
        Random rng = new Random(11);
        Conv2DLayer layer = new Conv2DLayer(2, 3, 3, 1, 1, Activation.SIGMOID, rng, () -> new SgdOptimizer(0.01));
        Tensor3D input = randomTensor(2, 6, 6, rng);
        Tensor3D upstreamGradient = randomTensor(3, 6, 6, rng);

        layer.forward(input);
        Tensor3D inputGradient = layer.backward(upstreamGradient);

        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 6; r++)
            {
                for (int c = 0; c < 6; c++)
                {
                    double analytical = inputGradient.get(ch, r, c);
                    double numerical = numericalInputGradient(layer, input, upstreamGradient, ch, r, c);
                    assertEquals(numerical, analytical, TOLERANCE, "ch=" + ch + " r=" + r + " c=" + c);
                }
            }
        }
    }

    @Test
    void gradientsMatchWithStrideAndNoPadding()
    {
        Random rng = new Random(17);
        Conv2DLayer layer = new Conv2DLayer(1, 2, 3, 2, 0, Activation.RELU, rng, () -> new SgdOptimizer(0.01));
        Tensor3D input = randomTensor(1, 7, 7, rng);
        int outSize = Conv2DLayer.computeOutputSize(7, 3, 2, 0);
        Tensor3D upstreamGradient = randomTensor(2, outSize, outSize, rng);

        layer.forward(input);
        layer.backward(upstreamGradient);

        for (int f = 0; f < 2; f++)
        {
            for (int kr = 0; kr < 3; kr++)
            {
                for (int kc = 0; kc < 3; kc++)
                {
                    double analytical = layer.getFilterGradient(f, 0, kr, kc);
                    double numerical = numericalFilterGradient(layer, input, upstreamGradient, f, 0, kr, kc);
                    assertEquals(numerical, analytical, TOLERANCE);
                }
            }
        }
    }

    private double loss(Conv2DLayer layer, Tensor3D input, Tensor3D upstreamGradient)
    {
        return dot(layer.forward(input), upstreamGradient);
    }

    private double dot(Tensor3D a, Tensor3D b)
    {
        double sum = 0;
        for (int ch = 0; ch < a.getChannels(); ch++)
        {
            for (int r = 0; r < a.getHeight(); r++)
            {
                for (int c = 0; c < a.getWidth(); c++)
                {
                    sum += a.get(ch, r, c) * b.get(ch, r, c);
                }
            }
        }
        return sum;
    }

    private Tensor3D randomTensor(int channels, int height, int width, Random rng)
    {
        Tensor3D t = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    t.set(ch, r, c, rng.nextDouble() * 2 - 1);
                }
            }
        }
        return t;
    }

    private double numericalFilterGradient(Conv2DLayer layer, Tensor3D input, Tensor3D upstreamGradient,
                                            int f, int c, int kr, int kc)
    {
        double original = layer.getFilterValue(f, c, kr, kc);

        layer.setFilterValue(f, c, kr, kc, original + EPSILON);
        double lossPlus = loss(layer, input, upstreamGradient);

        layer.setFilterValue(f, c, kr, kc, original - EPSILON);
        double lossMinus = loss(layer, input, upstreamGradient);

        layer.setFilterValue(f, c, kr, kc, original);
        return (lossPlus - lossMinus) / (2 * EPSILON);
    }

    private double numericalBiasGradient(Conv2DLayer layer, Tensor3D input, Tensor3D upstreamGradient, int f)
    {
        double original = layer.getBias(f);

        layer.setBias(f, original + EPSILON);
        double lossPlus = loss(layer, input, upstreamGradient);

        layer.setBias(f, original - EPSILON);
        double lossMinus = loss(layer, input, upstreamGradient);

        layer.setBias(f, original);
        return (lossPlus - lossMinus) / (2 * EPSILON);
    }

    private double numericalInputGradient(Conv2DLayer layer, Tensor3D input, Tensor3D upstreamGradient,
                                           int ch, int r, int c)
    {
        double original = input.get(ch, r, c);

        input.set(ch, r, c, original + EPSILON);
        double lossPlus = loss(layer, input, upstreamGradient);

        input.set(ch, r, c, original - EPSILON);
        double lossMinus = loss(layer, input, upstreamGradient);

        input.set(ch, r, c, original);
        return (lossPlus - lossMinus) / (2 * EPSILON);
    }
}
