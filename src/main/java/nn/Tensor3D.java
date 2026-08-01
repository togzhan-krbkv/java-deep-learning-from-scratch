package nn;

import java.util.function.DoubleUnaryOperator;

/**
 * Tensor3D, a channels x height x width array of doubles, used to
 * represent feature maps between convolutional layers.
 *
 * @author Togzhan K.
 */
public class Tensor3D
{
    private final double[][][] data;
    private final int channels;
    private final int height;
    private final int width;

    public Tensor3D(int channels, int height, int width)
    {
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.data = new double[channels][height][width];
    }

    public Tensor3D(double[][][] data)
    {
        this.channels = data.length;
        this.height = channels == 0 ? 0 : data[0].length;
        this.width = height == 0 ? 0 : data[0][0].length;
        this.data = data;
    }

    public int getChannels()
    {
        return channels;
    }

    public int getHeight()
    {
        return height;
    }

    public int getWidth()
    {
        return width;
    }

    public double get(int channel, int row, int col)
    {
        return data[channel][row][col];
    }

    public void set(int channel, int row, int col, double value)
    {
        data[channel][row][col] = value;
    }

    /** The given channel as a 2D Matrix slice */
    public Matrix getChannel(int channel)
    {
        Matrix result = new Matrix(height, width);
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                result.set(r, c, data[channel][r][c]);
            }
        }
        return result;
    }

    public Tensor3D add(Tensor3D other)
    {
        requireSameShape(other);
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = this.data[ch][r][c] + other.data[ch][r][c];
                }
            }
        }
        return result;
    }

    public Tensor3D subtract(Tensor3D other)
    {
        requireSameShape(other);
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = this.data[ch][r][c] - other.data[ch][r][c];
                }
            }
        }
        return result;
    }

    public Tensor3D hadamard(Tensor3D other)
    {
        requireSameShape(other);
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = this.data[ch][r][c] * other.data[ch][r][c];
                }
            }
        }
        return result;
    }

    public Tensor3D scale(double scalar)
    {
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = this.data[ch][r][c] * scalar;
                }
            }
        }
        return result;
    }

    public Tensor3D map(DoubleUnaryOperator fn)
    {
        Tensor3D result = new Tensor3D(channels, height, width);
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = fn.applyAsDouble(this.data[ch][r][c]);
                }
            }
        }
        return result;
    }

    /** Flattens to a (channels*height*width) x 1 column vector, channel-major order */
    public Matrix flatten()
    {
        Matrix result = new Matrix(channels * height * width, 1);
        int index = 0;
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.set(index, 0, data[ch][r][c]);
                    index++;
                }
            }
        }
        return result;
    }

    /** Inverse of flatten(): rebuilds a Tensor3D from a column vector in the same channel-major order */
    public static Tensor3D unflatten(Matrix column, int channels, int height, int width)
    {
        if (column.getRows() != channels * height * width || column.getCols() != 1)
        {
            throw new IllegalArgumentException(
                    "Expected a " + (channels * height * width) + "x1 column vector");
        }

        Tensor3D result = new Tensor3D(channels, height, width);
        int index = 0;
        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    result.data[ch][r][c] = column.get(index, 0);
                    index++;
                }
            }
        }
        return result;
    }

    public static Tensor3D zeros(int channels, int height, int width)
    {
        return new Tensor3D(channels, height, width);
    }

    /** Wraps a single Matrix as a one-channel Tensor3D */
    public static Tensor3D fromMatrix(Matrix matrix)
    {
        Tensor3D result = new Tensor3D(1, matrix.getRows(), matrix.getCols());
        for (int r = 0; r < matrix.getRows(); r++)
        {
            for (int c = 0; c < matrix.getCols(); c++)
            {
                result.data[0][r][c] = matrix.get(r, c);
            }
        }
        return result;
    }

    private void requireSameShape(Tensor3D other)
    {
        if (this.channels != other.channels || this.height != other.height || this.width != other.width)
        {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + channels + "x" + height + "x" + width
                            + " vs " + other.channels + "x" + other.height + "x" + other.width);
        }
    }
}
