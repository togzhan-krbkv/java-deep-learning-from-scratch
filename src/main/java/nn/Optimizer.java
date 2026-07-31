package nn;

/**
 * Applies a gradient update to a parameter matrix.
 *
 * @author Togzhan K.
 */
public interface Optimizer
{
    Matrix update(Matrix parameter, Matrix gradient);
}
