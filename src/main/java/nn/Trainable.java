package nn;

/**
 * Trainable, something with learnable parameters updated once per batch.
 *
 * @author Togzhan K.
 */
public interface Trainable
{
    void applyGradients(int batchSize);
}
