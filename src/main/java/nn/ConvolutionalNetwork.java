package nn;

import java.util.List;

/**
 * ConvolutionalNetwork, a convolutional front end feeding into a dense
 * back end: Tensor3DLayer stack, then FlattenLayer, then Layer stack,
 * then softmax with cross-entropy loss.
 *
 * @author Togzhan K.
 */
public class ConvolutionalNetwork
{
    private final List<Tensor3DLayer> convStack;
    private final FlattenLayer flatten;
    private final List<Layer> denseStack;

    public ConvolutionalNetwork(List<Tensor3DLayer> convStack, List<Layer> denseStack)
    {
        this.convStack = convStack;
        this.flatten = new FlattenLayer();
        this.denseStack = denseStack;
    }

    public Matrix predict(Tensor3D input)
    {
        Tensor3D features = input;
        for (Tensor3DLayer layer : convStack)
        {
            features = layer.forward(features);
        }

        Matrix output = flatten.forward(features);
        for (Layer layer : denseStack)
        {
            output = layer.forward(output);
        }

        return softmax(output);
    }

    public double trainOnExample(Tensor3D input, Matrix target)
    {
        Matrix output = predict(input);
        double loss = crossEntropyLoss(output, target);
        Matrix delta = output.subtract(target);

        for (int i = denseStack.size() - 1; i >= 0; i--)
        {
            delta = denseStack.get(i).backward(delta);
        }

        Tensor3D tensorDelta = flatten.backward(delta);

        for (int i = convStack.size() - 1; i >= 0; i--)
        {
            tensorDelta = convStack.get(i).backward(tensorDelta);
        }

        return loss;
    }

    public void applyGradients(int batchSize)
    {
        for (Layer layer : denseStack)
        {
            layer.applyGradients(batchSize);
        }
        for (Tensor3DLayer layer : convStack)
        {
            if (layer instanceof Trainable trainable)
            {
                trainable.applyGradients(batchSize);
            }
        }
    }

    private Matrix softmax(Matrix z)
    {
        int n = z.getRows();
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++)
        {
            max = Math.max(max, z.get(i, 0));
        }

        double sumExp = 0.0;
        double[] exp = new double[n];
        for (int i = 0; i < n; i++)
        {
            exp[i] = Math.exp(z.get(i, 0) - max);
            sumExp += exp[i];
        }

        Matrix result = new Matrix(n, 1);
        for (int i = 0; i < n; i++)
        {
            result.set(i, 0, exp[i] / sumExp);
        }
        return result;
    }

    private double crossEntropyLoss(Matrix predicted, Matrix target)
    {
        double loss = 0.0;
        for (int i = 0; i < predicted.getRows(); i++)
        {
            double p = Math.max(predicted.get(i, 0), 1e-12);
            loss -= target.get(i, 0) * Math.log(p);
        }
        return loss;
    }
}
