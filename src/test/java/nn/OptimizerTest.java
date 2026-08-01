package nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OptimizerTest
{
    @Test
    void sgdSubtractsScaledGradient()
    {
        Optimizer sgd = new SgdOptimizer(0.1);
        Matrix parameter = new Matrix(new double[][] { { 1.0 } });
        Matrix gradient = new Matrix(new double[][] { { 2.0 } });

        Matrix updated = sgd.update(parameter, gradient);

        assertEquals(0.8, updated.get(0, 0), 1e-9);
    }

    @Test
    void adamFirstStepMatchesHandComputedValue()
    {
        Optimizer adam = new AdamOptimizer(0.1);
        Matrix parameter = new Matrix(new double[][] { { 0.0 } });
        Matrix gradient = new Matrix(new double[][] { { 1.0 } });

        Matrix updated = adam.update(parameter, gradient);

        assertEquals(-0.099999999, updated.get(0, 0), 1e-6);
    }

    @Test
    void adamAccumulatesStateAcrossCalls()
    {
        Optimizer adam = new AdamOptimizer(0.1);
        Matrix parameter = new Matrix(new double[][] { { 0.0 } });
        Matrix gradient = new Matrix(new double[][] { { 1.0 } });

        Matrix firstUpdate = adam.update(parameter, gradient);
        Matrix secondUpdate = adam.update(firstUpdate, gradient);

        assertNotEquals(firstUpdate.get(0, 0) - parameter.get(0, 0),
                secondUpdate.get(0, 0) - firstUpdate.get(0, 0));
    }

    @Test
    void separateOptimizerInstancesTrackIndependentState()
    {
        Optimizer adamA = new AdamOptimizer(0.1);
        Optimizer adamB = new AdamOptimizer(0.1);
        Matrix parameter = new Matrix(new double[][] { { 0.0 } });
        Matrix gradient = new Matrix(new double[][] { { 1.0 } });

        adamA.update(parameter, gradient);
        Matrix bFirstUpdate = adamB.update(parameter, gradient);

        assertEquals(-0.099999999, bFirstUpdate.get(0, 0), 1e-6);
    }
}
