package nn;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class DenseLayerTest
{
    @Test
    void forwardProducesCorrectOutputShape()
    {
        DenseLayer layer = new DenseLayer(3, 5, Activation.RELU, new Random(1), () -> new SgdOptimizer(0.1));
        Matrix input = Matrix.columnVector(new double[] { 1, 2, 3 });

        Matrix output = layer.forward(input);

        assertEquals(5, output.getRows());
        assertEquals(1, output.getCols());
    }

    @Test
    void trainingReducesOutputErrorTowardTarget()
    {
        DenseLayer layer = new DenseLayer(2, 1, Activation.SIGMOID, new Random(1), () -> new SgdOptimizer(0.5));
        Matrix input = Matrix.columnVector(new double[] { 1, 1 });
        Matrix target = Matrix.columnVector(new double[] { 0.9 });

        double firstError = Math.abs(layer.forward(input).get(0, 0) - target.get(0, 0));

        for (int i = 0; i < 100; i++)
        {
            Matrix output = layer.forward(input);
            Matrix delta = output.subtract(target);
            layer.backward(delta);
            layer.applyGradients(1);
        }

        double lastError = Math.abs(layer.forward(input).get(0, 0) - target.get(0, 0));
        assertTrue(lastError < firstError);
    }

    @Test
    void batchGradientAveragingMatchesEquivalentSingleUpdate()
    {
        DenseLayer singleUpdateLayer = new DenseLayer(2, 1, Activation.LINEAR, new Random(7), () -> new SgdOptimizer(0.1));
        DenseLayer batchLayer = new DenseLayer(2, 1, Activation.LINEAR, new Random(7), () -> new SgdOptimizer(0.1));

        Matrix input = Matrix.columnVector(new double[] { 1, 2 });
        Matrix delta = Matrix.columnVector(new double[] { 0.5 });

        singleUpdateLayer.forward(input);
        singleUpdateLayer.backward(delta);
        singleUpdateLayer.applyGradients(1);

        batchLayer.forward(input);
        batchLayer.backward(delta);
        batchLayer.forward(input);
        batchLayer.backward(delta);
        batchLayer.applyGradients(2);

        Matrix probe = Matrix.columnVector(new double[] { 3, 4 });
        double expected = singleUpdateLayer.forward(probe).get(0, 0);
        double actual = batchLayer.forward(probe).get(0, 0);
        assertEquals(expected, actual, 1e-9);
    }
}
