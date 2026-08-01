package nn;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class ConvolutionalNetworkTest
{
    private static ConvolutionalNetwork buildTinyNetwork(Random rng)
    {
        java.util.function.Supplier<Optimizer> optimizerFactory = () -> new AdamOptimizer(0.01);

        List<Tensor3DLayer> convStack = new ArrayList<>();
        convStack.add(new Conv2DLayer(1, 4, 3, 1, 1, Activation.RELU, rng, optimizerFactory));
        convStack.add(new MaxPoolLayer(2, 2));

        List<Layer> denseStack = new ArrayList<>();
        denseStack.add(new DenseLayer(4 * 4 * 4, 3, Activation.LINEAR, rng, optimizerFactory));

        return new ConvolutionalNetwork(convStack, denseStack);
    }

    @Test
    void predictOutputsSumToOne()
    {
        ConvolutionalNetwork network = buildTinyNetwork(new Random(1));
        Tensor3D input = new Tensor3D(1, 8, 8);

        Matrix output = network.predict(input);

        assertEquals(3, output.getRows());
        double sum = 0;
        for (int i = 0; i < 3; i++)
        {
            assertTrue(output.get(i, 0) >= 0.0 && output.get(i, 0) <= 1.0);
            sum += output.get(i, 0);
        }
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void networkLearnsThreeDistinctImagePatterns()
    {
        Random rng = new Random(42);
        ConvolutionalNetwork network = buildTinyNetwork(rng);

        List<Tensor3D> images = List.of(
                patternImage("top-left"),
                patternImage("top-right"),
                patternImage("bottom"));
        List<Matrix> targets = List.of(
                Matrix.columnVector(new double[] { 1, 0, 0 }),
                Matrix.columnVector(new double[] { 0, 1, 0 }),
                Matrix.columnVector(new double[] { 0, 0, 1 }));
        int[] labels = { 0, 1, 2 };

        for (int epoch = 0; epoch < 300; epoch++)
        {
            for (int i = 0; i < images.size(); i++)
            {
                network.trainOnExample(images.get(i), targets.get(i));
            }
            network.applyGradients(images.size());
        }

        int correct = 0;
        for (int i = 0; i < images.size(); i++)
        {
            Matrix prediction = network.predict(images.get(i));
            if (prediction.argmax() == labels[i])
            {
                correct++;
            }
        }
        assertEquals(3, correct);
    }

    private static Tensor3D patternImage(String pattern)
    {
        Tensor3D image = new Tensor3D(1, 8, 8);
        switch (pattern)
        {
            case "top-left":
                for (int r = 0; r < 4; r++)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        image.set(0, r, c, 1.0);
                    }
                }
                break;
            case "top-right":
                for (int r = 0; r < 4; r++)
                {
                    for (int c = 4; c < 8; c++)
                    {
                        image.set(0, r, c, 1.0);
                    }
                }
                break;
            case "bottom":
                for (int r = 4; r < 8; r++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        image.set(0, r, c, 1.0);
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("unknown pattern: " + pattern);
        }
        return image;
    }
}
