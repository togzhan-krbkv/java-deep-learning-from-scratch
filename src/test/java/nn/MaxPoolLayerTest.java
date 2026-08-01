package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MaxPoolLayerTest
{
    @Test
    void forwardPicksMaxInEachNonOverlappingWindow()
    {
        Tensor3D input = new Tensor3D(1, 4, 4);
        double[][] values = {
                { 1, 3, 2, 4 },
                { 5, 6, 1, 2 },
                { 8, 1, 9, 0 },
                { 3, 2, 1, 5 }
        };
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                input.set(0, r, c, values[r][c]);
            }
        }

        MaxPoolLayer layer = new MaxPoolLayer(2, 2);
        Tensor3D output = layer.forward(input);

        assertEquals(2, output.getHeight());
        assertEquals(2, output.getWidth());
        assertEquals(6, output.get(0, 0, 0));
        assertEquals(4, output.get(0, 0, 1));
        assertEquals(8, output.get(0, 1, 0));
        assertEquals(9, output.get(0, 1, 1));
    }

    @Test
    void backwardRoutesGradientOnlyToMaxPosition()
    {
        Tensor3D input = new Tensor3D(1, 4, 4);
        double[][] values = {
                { 1, 3, 2, 4 },
                { 5, 6, 1, 2 },
                { 8, 1, 9, 0 },
                { 3, 2, 1, 5 }
        };
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                input.set(0, r, c, values[r][c]);
            }
        }

        MaxPoolLayer layer = new MaxPoolLayer(2, 2);
        layer.forward(input);

        Tensor3D upstream = new Tensor3D(1, 2, 2);
        upstream.set(0, 0, 0, 1);
        upstream.set(0, 0, 1, 2);
        upstream.set(0, 1, 0, 3);
        upstream.set(0, 1, 1, 4);

        Tensor3D inputGradient = layer.backward(upstream);

        assertEquals(1, inputGradient.get(0, 1, 1));
        assertEquals(2, inputGradient.get(0, 0, 3));
        assertEquals(3, inputGradient.get(0, 2, 0));
        assertEquals(4, inputGradient.get(0, 2, 2));

        assertEquals(0, inputGradient.get(0, 0, 0));
        assertEquals(0, inputGradient.get(0, 3, 3));
        assertEquals(0, inputGradient.get(0, 1, 0));
    }

    @Test
    void backwardAccumulatesGradientWhenWindowsOverlap()
    {
        Tensor3D input = new Tensor3D(1, 3, 3);
        double[][] values = {
                { 1, 2, 3 },
                { 4, 9, 5 },
                { 6, 7, 8 }
        };
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                input.set(0, r, c, values[r][c]);
            }
        }

        MaxPoolLayer layer = new MaxPoolLayer(2, 1);
        layer.forward(input);

        Tensor3D upstream = new Tensor3D(1, 2, 2).map(x -> 1.0);
        Tensor3D inputGradient = layer.backward(upstream);

        assertEquals(4, inputGradient.get(0, 1, 1));
        assertEquals(0, inputGradient.get(0, 0, 0));
        assertEquals(0, inputGradient.get(0, 2, 2));
    }

    @Test
    void channelsAreProcessedIndependently()
    {
        Tensor3D input = new Tensor3D(2, 2, 2);
        input.set(0, 0, 0, 1);
        input.set(0, 0, 1, 2);
        input.set(0, 1, 0, 3);
        input.set(0, 1, 1, 4);
        input.set(1, 0, 0, 40);
        input.set(1, 0, 1, 30);
        input.set(1, 1, 0, 20);
        input.set(1, 1, 1, 10);

        MaxPoolLayer layer = new MaxPoolLayer(2, 2);
        Tensor3D output = layer.forward(input);

        assertEquals(4, output.get(0, 0, 0));
        assertEquals(40, output.get(1, 0, 0));
    }
}
