package nn;

/**
 * MaxPoolLayer, downsamples each channel independently by taking the max
 * value in each poolSize x poolSize window. Has no learnable parameters;
 * the backward pass routes the gradient only to the position that was
 * the max in each window, summing contributions where windows overlap.
 *
 * @author Togzhan K.
 */
public class MaxPoolLayer implements Tensor3DLayer
{
    private final int poolSize;
    private final int stride;

    private Tensor3D lastInput;
    private int[][][] maxRow;
    private int[][][] maxCol;

    public MaxPoolLayer(int poolSize, int stride)
    {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public Tensor3D forward(Tensor3D input)
    {
        this.lastInput = input;
        int channels = input.getChannels();
        int outHeight = (input.getHeight() - poolSize) / stride + 1;
        int outWidth = (input.getWidth() - poolSize) / stride + 1;

        Tensor3D output = new Tensor3D(channels, outHeight, outWidth);
        maxRow = new int[channels][outHeight][outWidth];
        maxCol = new int[channels][outHeight][outWidth];

        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < outHeight; r++)
            {
                for (int c = 0; c < outWidth; c++)
                {
                    double max = Double.NEGATIVE_INFINITY;
                    int bestRow = -1;
                    int bestCol = -1;
                    for (int pr = 0; pr < poolSize; pr++)
                    {
                        for (int pc = 0; pc < poolSize; pc++)
                        {
                            int inRow = r * stride + pr;
                            int inCol = c * stride + pc;
                            double value = input.get(ch, inRow, inCol);
                            if (value > max)
                            {
                                max = value;
                                bestRow = inRow;
                                bestCol = inCol;
                            }
                        }
                    }
                    output.set(ch, r, c, max);
                    maxRow[ch][r][c] = bestRow;
                    maxCol[ch][r][c] = bestCol;
                }
            }
        }
        return output;
    }

    @Override
    public Tensor3D backward(Tensor3D dLossDOutput)
    {
        Tensor3D inputGradient = Tensor3D.zeros(
                lastInput.getChannels(), lastInput.getHeight(), lastInput.getWidth());

        int channels = dLossDOutput.getChannels();
        int outHeight = dLossDOutput.getHeight();
        int outWidth = dLossDOutput.getWidth();

        for (int ch = 0; ch < channels; ch++)
        {
            for (int r = 0; r < outHeight; r++)
            {
                for (int c = 0; c < outWidth; c++)
                {
                    int row = maxRow[ch][r][c];
                    int col = maxCol[ch][r][c];
                    double grad = dLossDOutput.get(ch, r, c);
                    inputGradient.set(ch, row, col, inputGradient.get(ch, row, col) + grad);
                }
            }
        }
        return inputGradient;
    }
}
