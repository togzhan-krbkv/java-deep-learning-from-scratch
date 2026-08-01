package nn;

/**
 * FlattenLayer, converts a Tensor3D feature map into a column vector
 * Matrix, and back on the way through backward. Remembers the input
 * shape from the last forward call so unflatten reconstructs it exactly.
 *
 * @author Togzhan K.
 */
public class FlattenLayer
{
    private int channels;
    private int height;
    private int width;

    public Matrix forward(Tensor3D input)
    {
        this.channels = input.getChannels();
        this.height = input.getHeight();
        this.width = input.getWidth();
        return input.flatten();
    }

    public Tensor3D backward(Matrix dLossDOutput)
    {
        return Tensor3D.unflatten(dLossDOutput, channels, height, width);
    }
}
