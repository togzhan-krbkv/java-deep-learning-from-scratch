package nn;

/**
 * Tensor3DLayer, one stage of a convolutional network: transforms a
 * feature map into another feature map.
 *
 * @author Togzhan K.
 */
public interface Tensor3DLayer
{
    Tensor3D forward(Tensor3D input);

    Tensor3D backward(Tensor3D gradient);
}
