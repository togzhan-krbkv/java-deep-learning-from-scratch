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
- Stochastic gradient descent, one example at a time, with the training
  set reshuffled each epoch

## Project structure

```
neural-network-java/
├── README.md
├── training_log.txt                    Full console output from the MNIST run below
├── pom.xml                             Maven project descriptor
├── src/main/java/nn/
│   ├── Matrix.java                     Dense matrix, all linear algebra
│   ├── Activation.java                 Sigmoid, ReLU, Linear + derivatives
│   ├── Layer.java                      One fully connected layer
│   ├── NeuralNetwork.java              Full network, forward and backward
│   ├── MnistLoader.java                Loads MNIST from CSV
│   ├── Trainer.java                    Epoch loop, shuffling, accuracy
│   └── Main.java                       Entry point
├── src/test/java/nn/
│   ├── MatrixTest.java                 8 unit tests for Matrix
│   └── NeuralNetworkTest.java          XOR convergence and softmax tests
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

**Weight update.** Standard gradient descent: for each weight,
`w = w - learningRate * gradient`. Learning rate is a fixed
hyperparameter (no adaptive optimizers like Adam here, on purpose, to
keep the implementation minimal and understandable).

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

- The XOR unit test trains a `2, 8, 1` network with sigmoid + MSE and
  reaches predictions within 1% of the correct 0/1 values. This is the
  classic proof that forward pass, backprop, and gradient descent are
  all implemented correctly, since XOR is not linearly separable and
  cannot be learned by any network without a hidden layer.
- Softmax outputs sum to 1.0 to within floating point tolerance.
- Cross-entropy loss decreases with training on a trivial three-class
  classification problem.

Trained end to end on the full MNIST dataset (60,000 training images,
10,000 test images) on a `784, 128, 10` architecture for 10 epochs,
learning rate 0.01:

| Epoch | Avg loss | Test accuracy |
|-------|----------|----------------|
| 1     | 0.2287   | 96.32%         |
| 2     | 0.0999   | 96.40%         |
| 3     | 0.0719   | 97.45%         |
| 4     | 0.0550   | 97.41%         |
| 5     | 0.0443   | 97.87%         |
| 6     | 0.0354   | 97.86%         |
| 7     | 0.0268   | 97.90%         |
| 8     | 0.0215   | 97.66%         |
| 9     | 0.0169   | 97.76%         |
| 10    | 0.0129   | **97.97%**     |

Full run output is in [training_log.txt](./training_log.txt). Loss drops
steadily and test accuracy climbs from 96.3% to 97.97%, in line with
published benchmarks for this architecture on MNIST.

## Why from scratch

Modern ML frameworks (PyTorch, TensorFlow, scikit-learn) abstract away
almost everything interesting about how a neural network actually
works. Implementing one by hand forces every design decision to become
explicit: how weights are laid out in memory, how gradients flow
backwards through layers, why softmax + cross-entropy is the natural
pairing, why He initialization matters for ReLU. That transparency is
the point of this project.
