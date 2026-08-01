package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Tensor3DTest
{
    @Test
    void getAndSetAddressCorrectElement()
    {
        Tensor3D t = new Tensor3D(2, 3, 4);
        t.set(1, 2, 3, 9.5);

        assertEquals(9.5, t.get(1, 2, 3));
        assertEquals(0.0, t.get(0, 2, 3));
        assertEquals(0.0, t.get(1, 1, 3));
    }

    @Test
    void dimensionsMatchConstructorArguments()
    {
        Tensor3D t = new Tensor3D(3, 5, 7);
        assertEquals(3, t.getChannels());
        assertEquals(5, t.getHeight());
        assertEquals(7, t.getWidth());
    }

    @Test
    void getChannelExtractsCorrectSlice()
    {
        Tensor3D t = new Tensor3D(2, 2, 2);
        t.set(1, 0, 0, 1);
        t.set(1, 0, 1, 2);
        t.set(1, 1, 0, 3);
        t.set(1, 1, 1, 4);

        Matrix channel = t.getChannel(1);

        assertEquals(1, channel.get(0, 0));
        assertEquals(2, channel.get(0, 1));
        assertEquals(3, channel.get(1, 0));
        assertEquals(4, channel.get(1, 1));
    }

    @Test
    void addAndSubtractAreInverses()
    {
        Tensor3D a = new Tensor3D(2, 2, 2).map(x -> 3.0);
        Tensor3D b = new Tensor3D(2, 2, 2).map(x -> 5.0);

        Tensor3D sum = a.add(b);
        Tensor3D backToA = sum.subtract(b);

        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    assertEquals(3.0, backToA.get(ch, r, c), 1e-9);
                }
            }
        }
    }

    @Test
    void scaleMultipliesEveryElement()
    {
        Tensor3D t = new Tensor3D(1, 2, 2).map(x -> 2.0);
        Tensor3D scaled = t.scale(4.0);

        assertEquals(8.0, scaled.get(0, 0, 0));
        assertEquals(8.0, scaled.get(0, 1, 1));
    }

    @Test
    void mapAppliesFunctionToEveryElement()
    {
        Tensor3D t = new Tensor3D(1, 2, 2).map(x -> 3.0);
        Tensor3D squared = t.map(x -> x * x);

        assertEquals(9.0, squared.get(0, 0, 0));
        assertEquals(9.0, squared.get(0, 1, 1));
    }

    @Test
    void flattenAndUnflattenRoundTrip()
    {
        Tensor3D original = new Tensor3D(2, 2, 2);
        double value = 0;
        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    original.set(ch, r, c, value);
                    value++;
                }
            }
        }

        Matrix flat = original.flatten();
        Tensor3D restored = Tensor3D.unflatten(flat, 2, 2, 2);

        for (int ch = 0; ch < 2; ch++)
        {
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    assertEquals(original.get(ch, r, c), restored.get(ch, r, c));
                }
            }
        }
    }

    @Test
    void flattenUsesChannelMajorOrder()
    {
        Tensor3D t = new Tensor3D(2, 1, 2);
        t.set(0, 0, 0, 10);
        t.set(0, 0, 1, 20);
        t.set(1, 0, 0, 30);
        t.set(1, 0, 1, 40);

        Matrix flat = t.flatten();

        assertEquals(4, flat.getRows());
        assertEquals(10, flat.get(0, 0));
        assertEquals(20, flat.get(1, 0));
        assertEquals(30, flat.get(2, 0));
        assertEquals(40, flat.get(3, 0));
    }

    @Test
    void unflattenRejectsWrongSizeColumn()
    {
        Matrix wrongSize = Matrix.columnVector(new double[] { 1, 2, 3 });
        assertThrows(IllegalArgumentException.class, () -> Tensor3D.unflatten(wrongSize, 2, 2, 2));
    }

    @Test
    void fromMatrixWrapsAsSingleChannel()
    {
        Matrix source = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        Tensor3D t = Tensor3D.fromMatrix(source);

        assertEquals(1, t.getChannels());
        assertEquals(2, t.getHeight());
        assertEquals(2, t.getWidth());
        assertEquals(1, t.get(0, 0, 0));
        assertEquals(4, t.get(0, 1, 1));
    }

    @Test
    void addThrowsOnShapeMismatch()
    {
        Tensor3D a = new Tensor3D(1, 2, 2);
        Tensor3D b = new Tensor3D(1, 3, 3);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
    }
}
