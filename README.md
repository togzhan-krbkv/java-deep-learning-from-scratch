# Neural Network in Java, from Scratch

A neural network library implemented from scratch in Java with no
machine learning libraries: a multilayer perceptron and a convolutional
network, sharing the same matrix, activation, optimizer, and dense layer
building blocks. Every matrix operation, convolution, activation
function, loss function, and gradient is written by hand, so the whole
training loop, from a raw image to an updated weight, is transparent
and inspectable.

Built as an independent portfolio project to demonstrate hands-on
understanding of neural network internals ahead of applying to Georgia
Tech's OMSCS program (AI specialization).

## What it does

Trains two kinds of networks to classify 28x28 grayscale images of
handwritten digits (the MNIST dataset):

**Multilayer perceptron** (`nn.Main`): a fully connected network with
ReLU hidden layers and a softmax output trained with cross-entropy loss.

**Convolutional network** (`nn.MainCnn`): Conv, Pool, Conv, Pool,
Flatten, Dense, Dense, then softmax, using the same dense layer,
activation, and optimizer code as the MLP.

Shared features:

- He-style weight initialization
- Mini-batch gradient descent, with gradients accumulated across a batch
  and applied once per batch
- Pluggable optimizers: plain SGD or Adam (adaptive moments with bias
  correction), each parameter tensor holding independent optimizer state
- Layers implement a common `Layer` interface (dense) or `Tensor3DLayer`
  interface (convolutional), both sharing a `Trainable` contract for
  batch-based gradient updates

## Project structure

```
neural-network-java/
├── README.md
├── training_log.txt                    MLP, SGD run, batch=1
├── training_log_adam.txt               MLP, Adam run, batch=32
├── training_log_cnn.txt                CNN, Adam run, batch=32
├── pom.xml                             Maven project descriptor
├── src/main/java/nn/
│   ├── Matrix.java                     Dense matrix, all linear algebra
│   ├── Tensor3D.java                   channels x height x width array for feature maps
│   ├── Activation.java                 Sigmoid, ReLU, Linear + derivatives
│   ├── Optimizer.java                  Strategy interface for parameter updates
│   ├── SgdOptimizer.java               Plain gradient descent
│   ├── AdamOptimizer.java              Adam with bias-corrected moments
│   ├── Trainable.java                  Interface: applyGradients(batchSize)
│   ├── Layer.java                      Interface: forward, backward (Matrix), extends Trainable
│   ├── DenseLayer.java                 Fully-connected Layer implementation
│   ├── Tensor3DLayer.java              Interface: forward, backward (Tensor3D)
│   ├── Conv2DLayer.java                Convolutional layer, implements Tensor3DLayer + Trainable
│   ├── MaxPoolLayer.java               Max pooling, implements Tensor3DLayer
│   ├── FlattenLayer.java               Bridges Tensor3D feature maps to Matrix vectors
│   ├── NeuralNetwork.java              MLP: a sequence of Layers, forward and backward
│   ├── ConvolutionalNetwork.java       CNN: Tensor3DLayer stack -> Flatten -> Layer stack
│   ├── MnistLoader.java                Loads MNIST from CSV (flattened Matrix and Tensor3D image)
│   ├── Trainer.java                    MLP mini-batch epoch loop, shuffling, accuracy
│   ├── CnnTrainer.java                 CNN mini-batch epoch loop, shuffling, accuracy
│   ├── Main.java                       MLP entry point
│   └── MainCnn.java                    CNN entry point
├── src/test/java/nn/
│   ├── MatrixTest.java                 8 unit tests for Matrix
│   ├── Tensor3DTest.java               11 unit tests for Tensor3D
│   ├── OptimizerTest.java              4 unit tests for SGD and Adam
│   ├── DenseLayerTest.java             3 unit tests for DenseLayer in isolation
│   ├── Conv2DLayerTest.java            8 hand-verified forward pass tests
│   ├── Conv2DGradientCheckTest.java    4 numerical gradient checking tests
│   ├── MaxPoolLayerTest.java           4 tests, including overlapping-window gradient accumulation
│   ├── FlattenLayerTest.java           3 round-trip and shape tests
│   ├── NeuralNetworkTest.java          XOR convergence (3 setups), softmax, loss tests
│   └── ConvolutionalNetworkTest.java   Softmax shape check, end-to-end learning on synthetic images
└── data/
    ├── mnist_train.csv                 Not tracked in git, see below
    └── mnist_test.csv                  Not tracked in git, see below
```

## The math, briefly

**Forward pass (dense).** For a single input column vector `x`, each
layer computes `a = activation(W x + b)`, and the result becomes the
input to the next layer. The final output layer either applies sigmoid
(for simple regression-style problems) or softmax across the whole
vector (for multi-class classification).

**Forward pass (convolution).** Each filter slides across the padded
input with a given stride; at each position, the output value is the
sum over input channels and kernel positions of `filter * input`, plus
a bias, then the activation function. Max pooling slides a window
across each channel independently and keeps only the maximum value,
remembering its position for the backward pass.

**Loss.** For softmax + cross-entropy classification with one-hot
target `y`, the loss for one example is `-sum(y_i * log(p_i))` where
`p` is the softmax output.

**Backward pass (dense).** Backpropagation is chain-rule gradient
computation running from the output layer back to the input. For
softmax + cross-entropy specifically, the derivative of the loss with
respect to the output layer's pre-activation simplifies neatly to
`predicted - target`. The output layer's own activation is `LINEAR`
(softmax is applied separately, across the whole output vector), and
`LINEAR`'s derivative is always 1, so the same `backward()` used by
every other layer handles this case correctly without a special path.

**Backward pass (convolution).** Each filter's weight gradient is the
sum, over every output position, of the upstream gradient at that
position times the input patch it multiplied during the forward pass.
The bias gradient is the sum of the upstream gradient over all output
positions. The gradient flowing back to the input accumulates, at each
input position, the filter weight times the upstream gradient, summed
over every output position that touched it (positions overlap when
stride is smaller than the kernel size). Max pooling routes the
gradient only to the position that was the maximum in each window,
summing when windows overlap. Every formula here was independently
verified two ways: first with a small NumPy prototype checked against
finite differences, then in the actual Java implementation via
numerical gradient checking (see Results).

**Weight update.** Every layer accumulates gradients across a batch,
then hands the averaged gradient to an `Optimizer`. `SgdOptimizer`
applies plain gradient descent, `w = w - learningRate * gradient`.
`AdamOptimizer` tracks a running mean and variance of the gradient
(`m` and `v`), bias-corrects them, and scales the step by
`1 / (sqrt(v) + epsilon)`. Convolutional filters reuse the same
`Optimizer` implementations as dense layers by flattening each filter
into a column vector, applying the update, and unflattening it back
into a `Tensor3D`.

## Layer architecture

Two small interfaces cover everything: `Layer` (`forward`/`backward` on
`Matrix`, for dense layers) and `Tensor3DLayer` (`forward`/`backward`
on `Tensor3D`, for convolutional and pooling layers). Both share a
`Trainable` contract (`applyGradients(batchSize)`) for anything with
learnable parameters; `MaxPoolLayer` has none, so it implements only
`Tensor3DLayer`.

`ConvolutionalNetwork` composes the two halves: a `List<Tensor3DLayer>`
front end, a `FlattenLayer` bridge (backed by `Tensor3D.flatten()` and
`unflatten()`), and a `List<Layer>` back end, ending in the same
softmax and cross-entropy math as the plain MLP. It does not reuse
`NeuralNetwork` directly, since that class was not designed to expose
its internal gradient at the input boundary; the roughly 15 lines of
softmax and cross-entropy logic are duplicated rather than risking
changes to the already-tested `NeuralNetwork` class.

Earlier, `Layer` had two backward methods: a general one, and a second
used only for the output layer under softmax + cross-entropy, to skip
multiplying by an activation derivative. That distinction turned out to
be unnecessary, since the output layer's activation is already `LINEAR`
when softmax is used, and `LINEAR`'s derivative is always 1. Removing
the special case made `Layer` a strictly smaller, cleaner interface.

## How to build and run

```
mvn test                        # run all unit tests
mvn compile                     # compile the main sources
java -cp target/classes nn.Main       # train the MLP
java -cp target/classes nn.MainCnn    # train the CNN
```

Both entry points expect two CSV files:

- `data/mnist_train.csv`
- `data/mnist_test.csv`

Each row has 785 columns: a digit label (0-9), then 784 pixel values
(0-255) for the flattened 28x28 image. The first row is a header. This
matches the [Kaggle MNIST CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
directly. The CSV files are not included in this repository (the
training file alone is over 100MB); download them from Kaggle and
place them in `data/` before running.

## Results

Verified sanity checks (all pass):

- The XOR unit test converges under three different optimization setups:
  plain SGD updating after every example, full-batch SGD, and Adam. This
  is the classic proof that forward pass, backprop, and gradient descent
  are all implemented correctly, since XOR is not linearly separable and
  cannot be learned by any network without a hidden layer.
- Adam's first update step matches a hand-computed value, and its
  internal moment estimates persist correctly across calls.
- Softmax outputs sum to 1.0 to within floating point tolerance.
- Cross-entropy loss decreases with training on a trivial three-class
  classification problem.
- Convolution forward pass output matches hand-computed values under no
  padding, padding, stride, bias, multiple input channels, and multiple
  filters.
- Convolution backward pass gradients (filters, biases, and input) match
  numerical gradients from finite differences, checked at every weight
  in a small test layer, both with and without padding and stride.
  Before implementing this in Java, the same formulas were independently
  verified in a NumPy prototype, where the numerical and analytical
  gradients agreed to within 1e-10.
- Max pooling's backward pass correctly sums the gradient when multiple
  overlapping windows share the same maximum position.
- A tiny end-to-end convolutional network (conv, pool, flatten, dense)
  learns to correctly classify three distinct synthetic image patterns.

Trained end to end on the full MNIST dataset (60,000 training images,
10,000 test images) for 10 epochs, batch size 32, Adam, learning rate
0.001:

| Epoch | MLP (784, 128, 10) | CNN (Conv-Pool-Conv-Pool-Dense-Dense) |
|-------|---------------------|------------------------------------------|
| 1     | 95.23%              | 97.41%                                    |
| 2     | 96.70%              | 98.27%                                    |
| 3     | 97.30%              | 98.61%                                    |
| 4     | 97.36%              | 98.57%                                    |
| 5     | 97.74%              | 98.70%                                    |
| 6     | 97.84%              | **98.89%**                                |
| 7     | 97.72%              | 98.86%                                    |
| 8     | 97.69%              | 98.84%                                    |
| 9     | 97.81%              | 98.73%                                    |
| 10    | 97.79%              | 98.72%                                    |

Full run output is in [training_log_adam.txt](./training_log_adam.txt)
(MLP) and [training_log_cnn.txt](./training_log_cnn.txt) (CNN); an
earlier plain-SGD MLP run is in [training_log.txt](./training_log.txt).

The CNN reaches a higher accuracy than the MLP at every epoch, and cuts
the final error rate from about 2.21% to about 1.28%, a roughly 42%
relative reduction. This matches the expected result: convolution
explicitly reuses spatial structure (nearby pixels), while the MLP
treats all 784 pixels as an unordered list. The CNN is also
substantially slower per epoch (roughly 127-139 seconds versus about
75 seconds for the MLP), since neither `Conv2DLayer` nor `DenseLayer`
vectorizes across a batch. Both process one example at a time; a batch
only changes how often gradients are applied, not how much computation
each epoch does. Vectorizing forward and backward across a whole batch
as a single matrix multiplication, the way real frameworks do, remains
a natural next step.

## Why from scratch

Modern ML frameworks (PyTorch, TensorFlow, scikit-learn) abstract away
almost everything interesting about how a neural network actually
works. Implementing one by hand forces every design decision to become
explicit: how weights are laid out in memory, how gradients flow
backwards through layers, why softmax + cross-entropy is the natural
pairing, why He initialization matters for ReLU, and why a convolution's
backward pass has to account for overlapping windows. That transparency
is the point of this project.
