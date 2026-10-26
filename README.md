# MNIST Parser for Rust 

A simple, fast loader for the [MNIST dataset](https://yann.lecun.org/exdb/mnist/index.html) of handwritten digits.

## Features

* Correct & Safe: Parses the IDX file format according to the official specification.
* Ergonomic API: Provides a simple Image struct with helpful methods (as_u8_array, as_f32_array, as_f64_array, etc.).
* Expressive Errors: Uses a custom MnistError enum for precise, programmatic error handling.
* ASCII Visualization: Implements the Display trait on Image for quick terminal-based viewing.
* Optional Plotting: Includes a plotting feature flag that enables graphical display of images via gnuplot.
* Lightweight: Has zero dependencies by default. gnuplot is only included when the plotting feature is enabled.

## Getting Started
Follow these steps to get the data and run the included demo application.

### 1. Clone the Repository

```bash
git clone https://github.com/jesper-olsen/mnist-c.git
cd mnist-c
```
### 2. Download the MNIST Dataset

The original download links on Yann LeCun's website are often unavailable. The recommended way to get the data is from the TensorFlow/Keras datasets mirror.

Create a directory for the data and download the four files into it:

```bash
mkdir MNIST

curl -o MNIST/train-images-idx3-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
curl -o MNIST/train-labels-idx1-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
curl -o MNIST/t10k-images-idx3-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
curl -o MNIST/t10k-labels-idx1-ubyte.gz https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
```

Optional - download [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) also (same format):
```
mkdir MNISTfashion
curl -o MNISTfashion/train-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
curl -o MNISTfashion/train-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
curl -o MNISTfashion/t10k-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
curl -o MNISTfashion/t10k-labels-idx1-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```
Next, decompress the files:

``` bash
gunzip MNIST/*.gz
```

Your MNIST directory should now contain the four uncompressed data files.

### 3. Run the Demo Application
The repository includes a demo binary that uses this library to display images.

To show an image as ASCII art in the terminal:
(This command displays image #2 from the training set)

```bash
./demo MNIST train 2
```

Example output:
```text
--- Dataset: train | Image #2 | Label: 4 ---





                    :@.
    .:              :*.
    :*             .*@.
    @*             .@*
    @*             *@:
   .@*             @@.
   :@*            .@@.
   *@:            *@@.
   *@:         .:*@@:
   *@:   ..:**@@@*@@.
   *@@@@@@@@@@*:..@@
    :*****:.     :@@
                 *@*
                 *@.
                 *@.
                 *@:
                 *@:
                 *@*
                 *@*
                 :@*


```


## Demo Application Command-Line Arguments

```bash
./demo
usage: ./demo <data-dir> <train|test> <index>
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Yann LeCun, Corinna Cortes, and Christopher J.C. Burges for creating and curating the MNIST dataset.
