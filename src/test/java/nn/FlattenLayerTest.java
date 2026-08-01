package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FlattenLayerTest
{
    @Test
    void forwardFlattensToColumnVector()
    {
        Tensor3D input = new Tensor3D(2, 2, 2);
        double value = 0;
        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    input.set(ch, r, c, value);
                    value++;
                }
            }
        }

        FlattenLayer layer = new FlattenLayer();
        Matrix output = layer.forward(input);

        assertEquals(8, output.getRows());
        assertEquals(1, output.getCols());
        assertEquals(0, output.get(0, 0));
        assertEquals(7, output.get(7, 0));
    }

    @Test
    void backwardReconstructsOriginalShapeAndValues()
    {
        Tensor3D input = new Tensor3D(2, 3, 4);
        double value = 0;
        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    input.set(ch, r, c, value);
                    value++;
                }
            }
        }

        FlattenLayer layer = new FlattenLayer();
        Matrix flat = layer.forward(input);
        Tensor3D restored = layer.backward(flat);

        assertEquals(2, restored.getChannels());
        assertEquals(3, restored.getHeight());
        assertEquals(4, restored.getWidth());
        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    assertEquals(input.get(ch, r, c), restored.get(ch, r, c));
                }
            }
        }
    }

    @Test
    void backwardRejectsWrongSizeGradient()
    {
        FlattenLayer layer = new FlattenLayer();
        layer.forward(new Tensor3D(2, 2, 2));

        Matrix wrongSize = Matrix.columnVector(new double[] { 1, 2, 3 });
        assertThrows(IllegalArgumentException.class, () -> layer.backward(wrongSize));
    }
}
