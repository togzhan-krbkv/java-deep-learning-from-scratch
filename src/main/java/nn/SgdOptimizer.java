package nn;

/**
 * SgdOptimizer, plain gradient descent.
 *
 * @author Togzhan K.
 */
public class SgdOptimizer implements Optimizer
{
    private final double learningRate;

    public SgdOptimizer(double learningRate)
    {
        this.learningRate = learningRate;
    }

    @Override
    public Matrix update(Matrix parameter, Matrix gradient)
    {
        return parameter.subtract(gradient.scale(learningRate));
    }
}
