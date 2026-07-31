package nn;

/**
 * Activation, common activation functions used between layers, each
 * paired with its derivative for use during backpropagation.
 *
 * @author Togzhan K.
 */
public enum Activation
{
    SIGMOID
    {
        @Override
        public double apply(double x)
        {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double activatedValue)
        {
            // activatedValue is sigmoid(x); sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            return activatedValue * (1.0 - activatedValue);
        }
    },

    RELU
    {
        @Override
        public double apply(double x)
        {
            return Math.max(0.0, x);
        }

        @Override
        public double derivative(double activatedValue)
        {
            return activatedValue > 0.0 ? 1.0 : 0.0;
        }
    },

    LINEAR
    {
        @Override
        public double apply(double x)
        {
            return x;
        }

        @Override
        public double derivative(double activatedValue)
        {
            return 1.0;
        }
    };

    /** The activation function itself, applied to a pre-activation value */
    public abstract double apply(double x);

    /** Derivative expressed in terms of the already-activated output */
    public abstract double derivative(double activatedValue);
}
