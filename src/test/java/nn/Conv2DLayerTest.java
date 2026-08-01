package nn;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class Conv2DLayerTest
{
    private static Tensor3D makeInput4x4()
    {
        Tensor3D input = new Tensor3D(1, 4, 4);
        double value = 1;
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                input.set(0, r, c, value);
                value++;
            }
        }
        return input;
    }

    private static Tensor3D onesFilter(int channels, int size)
    {
        return new Tensor3D(channels, size, size).map(x -> 1.0);
    }

    @Test
    void forwardMatchesHandComputedValuesNoPaddingStrideOne()
    {
        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(1, 3) }, new double[] { 0 }, 1, 0, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(makeInput4x4());

        assertEquals(1, output.getChannels());
        assertEquals(2, output.getHeight());
        assertEquals(2, output.getWidth());
        assertEquals(54, output.get(0, 0, 0));
        assertEquals(63, output.get(0, 0, 1));
        assertEquals(90, output.get(0, 1, 0));
        assertEquals(99, output.get(0, 1, 1));
    }

    @Test
    void forwardMatchesHandComputedValuesWithPadding()
    {
        Tensor3D input = new Tensor3D(1, 2, 2);
        input.set(0, 0, 0, 1);
        input.set(0, 0, 1, 2);
        input.set(0, 1, 0, 3);
        input.set(0, 1, 1, 4);

        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(1, 2) }, new double[] { 0 }, 1, 1, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(input);

        assertEquals(3, output.getHeight());
        assertEquals(3, output.getWidth());
        double[][] expected = { { 1, 3, 2 }, { 4, 10, 6 }, { 3, 7, 4 } };
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                assertEquals(expected[r][c], output.get(0, r, c));
            }
        }
    }

    @Test
    void forwardMatchesHandComputedValuesWithStrideTwo()
    {
        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(1, 2) }, new double[] { 0 }, 2, 0, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(makeInput4x4());

        assertEquals(2, output.getHeight());
        assertEquals(2, output.getWidth());
        assertEquals(14, output.get(0, 0, 0));
        assertEquals(22, output.get(0, 0, 1));
        assertEquals(46, output.get(0, 1, 0));
        assertEquals(54, output.get(0, 1, 1));
    }

    @Test
    void biasIsAddedToEveryOutputPosition()
    {
        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(1, 3) }, new double[] { 100 }, 1, 0, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(makeInput4x4());

        assertEquals(154, output.get(0, 0, 0));
        assertEquals(199, output.get(0, 1, 1));
    }

    @Test
    void multipleInputChannelsAreSummed()
    {
        Tensor3D input = new Tensor3D(2, 2, 2);
        input.set(0, 0, 0, 1);
        input.set(0, 0, 1, 1);
        input.set(0, 1, 0, 1);
        input.set(0, 1, 1, 1);
        input.set(1, 0, 0, 10);
        input.set(1, 0, 1, 10);
        input.set(1, 1, 0, 10);
        input.set(1, 1, 1, 10);

        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(2, 2) }, new double[] { 0 }, 1, 0, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(input);

        assertEquals(1, output.getHeight());
        assertEquals(1, output.getWidth());
        assertEquals(44, output.get(0, 0, 0));
    }

    @Test
    void multipleFiltersProduceIndependentOutputChannels()
    {
        Conv2DLayer layer = new Conv2DLayer(
                new Tensor3D[] { onesFilter(1, 3), onesFilter(1, 3).scale(2) },
                new double[] { 0, 0 }, 1, 0, Activation.LINEAR, () -> new SgdOptimizer(0.01));

        Tensor3D output = layer.forward(makeInput4x4());

        assertEquals(2, output.getChannels());
        assertEquals(54, output.get(0, 0, 0));
        assertEquals(108, output.get(1, 0, 0));
    }

    @Test
    void computeOutputSizeMatchesActualForwardOutput()
    {
        int inputSize = 8;
        int kernelSize = 3;
        int stride = 2;
        int padding = 1;

        int expected = Conv2DLayer.computeOutputSize(inputSize, kernelSize, stride, padding);

        Conv2DLayer layer = new Conv2DLayer(1, 1, kernelSize, stride, padding, Activation.LINEAR, new Random(1), () -> new SgdOptimizer(0.01));
        Tensor3D output = layer.forward(new Tensor3D(1, inputSize, inputSize));

        assertEquals(expected, output.getHeight());
        assertEquals(expected, output.getWidth());
    }

    @Test
    void randomlyInitializedFiltersProduceCorrectOutputShape()
    {
        Conv2DLayer layer = new Conv2DLayer(3, 8, 5, 1, 2, Activation.RELU, new Random(42), () -> new SgdOptimizer(0.01));
        Tensor3D input = new Tensor3D(3, 28, 28);

        Tensor3D output = layer.forward(input);

        assertEquals(8, output.getChannels());
        assertEquals(28, output.getHeight());
        assertEquals(28, output.getWidth());
    }
}
