package nn;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Matrix, a dense matrix of doubles with the linear algebra operations
 * needed for a neural network.
 *
 * @author Togzhan K.
 */
public class Matrix
{
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols)
    {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data)
    {
        this.rows = data.length;
        this.cols = data.length == 0 ? 0 : data[0].length;
        this.data = data;
    }

    public int getRows()
    {
        return rows;
    }

    public int getCols()
    {
        return cols;
    }

    public double get(int r, int c)
    {
        return data[r][c];
    }

    public void set(int r, int c, double value)
    {
        data[r][c] = value;
    }

    /** Standard matrix multiplication: (rows x cols) * (cols x otherCols) */
    public Matrix multiply(Matrix other)
    {
        if (this.cols != other.rows)
        {
            throw new IllegalArgumentException(
                    "Cannot multiply " + rows + "x" + cols + " by " + other.rows + "x" + other.cols);
        }

        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < other.cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < this.cols; k++)
                {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public Matrix add(Matrix other)
    {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other)
    {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    /** Element-wise (Hadamard) product, used when applying activation derivatives during backprop */
    public Matrix hadamard(Matrix other)
    {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public Matrix scale(double scalar)
    {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    public Matrix transpose()
    {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    /** Applies fn to every element, returning a new Matrix (used for activation functions) */
    public Matrix map(DoubleUnaryOperator fn)
    {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = fn.applyAsDouble(this.data[i][j]);
            }
        }
        return result;
    }

    /** Sum of all elements, used for computing bias gradients */
    public double sum()
    {
        double total = 0.0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                total += data[i][j];
            }
        }
        return total;
    }

    /** Index of the largest value in a column vector (used to read off a predicted class) */
    public int argmax()
    {
        int bestRow = 0;
        double bestVal = data[0][0];
        for (int i = 1; i < rows; i++)
        {
            if (data[i][0] > bestVal)
            {
                bestVal = data[i][0];
                bestRow = i;
            }
        }
        return bestRow;
    }

    public static Matrix zeros(int rows, int cols)
    {
        return new Matrix(rows, cols);
    }

    /** Random initialization scaled by the given factor (e.g. Xavier/He-style scaling) */
    public static Matrix random(int rows, int cols, double scale, Random rng)
    {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m.data[i][j] = (rng.nextDouble() * 2 - 1) * scale;
            }
        }
        return m;
    }

    /** Wraps a 1D array as a column vector (rows x 1) */
    public static Matrix columnVector(double[] values)
    {
        Matrix m = new Matrix(values.length, 1);
        for (int i = 0; i < values.length; i++)
        {
            m.data[i][0] = values[i];
        }
        return m;
    }

    private void requireSameShape(Matrix other)
    {
        if (this.rows != other.rows || this.cols != other.cols)
        {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + rows + "x" + cols + " vs " + other.rows + "x" + other.cols);
        }
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sb.append(String.format("%.4f", data[i][j]));
                if (j < cols - 1)
                {
                    sb.append(", ");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
