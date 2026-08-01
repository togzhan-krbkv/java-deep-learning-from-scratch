package nn;

/**
 * Layer, one stage of a neural network: transforms an input into an
 * output, and accumulates gradients across a batch for later update.
 *
 * @author Togzhan K.
 */
public interface Layer extends Trainable
{
    Matrix forward(Matrix input);

    Matrix backward(Matrix gradient);
}
