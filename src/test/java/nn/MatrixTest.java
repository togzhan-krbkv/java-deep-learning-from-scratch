package nn;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class MatrixTest
{
    @Test
    void multiplyProducesCorrectShapeAndValues()
    {
        Matrix a = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        Matrix b = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });

        Matrix result = a.multiply(b);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(19, result.get(0, 0));
        assertEquals(22, result.get(0, 1));
        assertEquals(43, result.get(1, 0));
        assertEquals(50, result.get(1, 1));
    }

    @Test
    void multiplyThrowsOnIncompatibleShapes()
    {
        Matrix a = new Matrix(2, 3);
        Matrix b = new Matrix(2, 2);
        assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
    }

    @Test
    void addAndSubtractAreInverses()
    {
        Matrix a = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        Matrix b = new Matrix(new double[][] { { 5, 6 }, { 7, 8 } });

        Matrix sum = a.add(b);
        Matrix backToA = sum.subtract(b);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                assertEquals(a.get(i, j), backToA.get(i, j), 1e-9);
            }
        }
    }

    @Test
    void transposeSwapsDimensionsAndValues()
    {
        Matrix a = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        Matrix t = a.transpose();

        assertEquals(3, t.getRows());
        assertEquals(2, t.getCols());
        assertEquals(1, t.get(0, 0));
        assertEquals(4, t.get(0, 1));
        assertEquals(2, t.get(1, 0));
        assertEquals(6, t.get(2, 1));
    }

    @Test
    void hadamardMultipliesElementWise()
    {
        Matrix a = new Matrix(new double[][] { { 2, 3 }, { 4, 5 } });
        Matrix b = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });

        Matrix result = a.hadamard(b);

        assertEquals(2, result.get(0, 0));
        assertEquals(6, result.get(0, 1));
        assertEquals(12, result.get(1, 0));
        assertEquals(20, result.get(1, 1));
    }

    @Test
    void mapAppliesFunctionToEveryElement()
    {
        Matrix a = new Matrix(new double[][] { { 1, 2 }, { 3, 4 } });
        Matrix doubled = a.map(x -> x * 2);

        assertEquals(2, doubled.get(0, 0));
        assertEquals(4, doubled.get(0, 1));
        assertEquals(6, doubled.get(1, 0));
        assertEquals(8, doubled.get(1, 1));
    }

    @Test
    void argmaxFindsIndexOfLargestValue()
    {
        Matrix column = Matrix.columnVector(new double[] { 0.1, 0.7, 0.2 });
        assertEquals(1, column.argmax());
    }

    @Test
    void randomMatrixStaysWithinScaleBounds()
    {
        Matrix m = Matrix.random(10, 10, 0.5, new Random(42));
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                assertTrue(Math.abs(m.get(i, j)) <= 0.5);
            }
        }
    }
}
