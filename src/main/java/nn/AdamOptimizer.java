package nn;

/**
 * AdamOptimizer, adaptive moment estimation with bias correction.
 *
 * @author Togzhan K.
 */
public class AdamOptimizer implements Optimizer
{
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private Matrix m;
    private Matrix v;
    private int t;

    public AdamOptimizer(double learningRate)
    {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon)
    {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public Matrix update(Matrix parameter, Matrix gradient)
    {
        if (m == null)
        {
            m = Matrix.zeros(gradient.getRows(), gradient.getCols());
            v = Matrix.zeros(gradient.getRows(), gradient.getCols());
        }

        t++;
        m = m.scale(beta1).add(gradient.scale(1 - beta1));
        v = v.scale(beta2).add(gradient.hadamard(gradient).scale(1 - beta2));

        Matrix mHat = m.scale(1.0 / (1 - Math.pow(beta1, t)));
        Matrix vHat = v.scale(1.0 / (1 - Math.pow(beta2, t)));

        Matrix step = mHat.hadamard(vHat.map(x -> 1.0 / (Math.sqrt(x) + epsilon))).scale(learningRate);
        return parameter.subtract(step);
    }
}
