# Neural Network in Java, from Scratch

A multilayer perceptron with backpropagation, implemented from scratch in
Java with no machine learning libraries. Every matrix operation,
activation function, loss function, and gradient is written by hand, so
the whole training loop, from a raw image to an updated weight, is
transparent and inspectable.

Built as an independent portfolio project to demonstrate hands-on
understanding of neural network internals ahead of applying to Georgia
Tech's OMSCS program (AI specialization).

## What it does

Trains a fully connected neural network to classify 28x28 grayscale
images of handwritten digits (the MNIST dataset). The network is a
standard multilayer perceptron with:

- ReLU activation in the hidden layer
- Softmax output layer trained with cross-entropy loss
- He-style weight initialization
- Mini-batch gradient descent, with gradients accumulated across a batch
  and applied once per batch
- Pluggable optimizers: plain SGD or Adam (adaptive moments with bias
  correction), each layer holding independent optimizer state per
  parameter

## Project structure

```
neural-network-java/
├── README.md
├── training_log.txt                    SGD run, batch=1
├── training_log_adam.txt               Adam run, batch=32
├── pom.xml                             Maven project descriptor
├── src/main/java/nn/
│   ├── Matrix.java                     Dense matrix, all linear algebra
│   ├── Activation.java                 Sigmoid, ReLU, Linear + derivatives
│   ├── Optimizer.java                  Strategy interface for parameter updates
│   ├── SgdOptimizer.java               Plain gradient descent
│   ├── AdamOptimizer.java              Adam with bias-corrected moments
│   ├── Layer.java                      One fully connected layer
│   ├── NeuralNetwork.java              Full network, forward and backward
│   ├── MnistLoader.java                Loads MNIST from CSV
│   ├── Trainer.java                    Mini-batch epoch loop, shuffling, accuracy
│   └── Main.java                       Entry point
├── src/test/java/nn/
│   ├── MatrixTest.java                 8 unit tests for Matrix
│   ├── OptimizerTest.java              4 unit tests for SGD and Adam
│   └── NeuralNetworkTest.java          XOR convergence (3 setups), softmax, loss tests
└── data/
    ├── mnist_train.csv                 Not tracked in git, see below
    └── mnist_test.csv                  Not tracked in git, see below
```

## The math, briefly

**Forward pass.** For a single input column vector `x`, each layer
computes `a = activation(W x + b)`, and the result becomes the input to
the next layer. The final output layer either applies sigmoid (for
simple regression-style problems) or softmax across the whole vector
(for multi-class classification).

**Loss.** For softmax + cross-entropy classification with one-hot
target `y`, the loss for one example is `-sum(y_i * log(p_i))` where
`p` is the softmax output.

**Backward pass.** Backpropagation is chain-rule gradient computation
running from the output layer back to the input. For softmax +
cross-entropy specifically, the derivative of the loss with respect to
the layer's pre-activation simplifies neatly to `predicted - target`,
so that specific gradient is computed directly rather than by
multiplying by an activation derivative (which is why `Layer` has both
`backward` and `backwardWithPrecomputedDelta`). Every other layer uses
the standard chain rule: multiply the incoming gradient by the layer's
activation derivative, then compute the weight and bias gradients from
the cached input.

**Weight update.** Each layer accumulates a weight gradient and a bias
gradient across a batch, then hands the averaged gradient to an
`Optimizer`. `SgdOptimizer` applies plain gradient descent,
`w = w - learningRate * gradient`. `AdamOptimizer` tracks a running
mean and variance of the gradient (`m` and `v`), bias-corrects them,
and scales the step by `1 / (sqrt(v) + epsilon)`, adapting the
effective learning rate per parameter. Each layer holds independent
optimizer instances for its weights and biases, so their moment
estimates never mix.

## How to build and run

```
mvn test                    # run the unit tests (Matrix + NeuralNetwork)
mvn compile                 # compile the main sources
java -cp target/classes nn.Main
```

The training entry point expects two CSV files:

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

Trained end to end on the full MNIST dataset (60,000 training images,
10,000 test images) on a `784, 128, 10` architecture for 10 epochs,
under two configurations:

| Epoch | SGD, batch=1, lr=0.01 | Adam, batch=32, lr=0.001 |
|-------|------------------------|---------------------------|
| 1     | 96.32%                 | 95.23%                    |
| 2     | 96.40%                 | 96.70%                    |
| 3     | 97.45%                 | 97.30%                    |
| 4     | 97.41%                 | 97.36%                    |
| 5     | 97.87%                 | 97.74%                    |
| 6     | 97.86%                 | 97.84%                    |
| 7     | 97.90%                 | 97.72%                    |
| 8     | 97.66%                 | 97.69%                    |
| 9     | 97.76%                 | 97.81%                    |
| 10    | **97.97%**             | **97.79%**                |

Full run output for both is in [training_log.txt](./training_log.txt)
and [training_log_adam.txt](./training_log_adam.txt).

Adam with batch size 32 does not train faster per epoch here, and
reaches a very slightly lower final accuracy than plain per-example SGD.
This is expected given the implementation: `Layer.forward`/`backward`
still processes one example at a time, so batching only changes how
often gradients are applied (1,875 updates per epoch instead of
60,000), not how much computation each epoch does. Mini-batching here
gives its usual statistical benefit (averaged, less noisy gradients)
but not the computational speedup that real frameworks get from
vectorizing the whole batch into a single matrix multiplication. That
vectorization is a natural next step.

## Why from scratch

Modern ML frameworks (PyTorch, TensorFlow, scikit-learn) abstract away
almost everything interesting about how a neural network actually
works. Implementing one by hand forces every design decision to become
explicit: how weights are laid out in memory, how gradients flow
backwards through layers, why softmax + cross-entropy is the natural
pairing, why He initialization matters for ReLU. That transparency is
the point of this project.
