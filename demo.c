#define MNIST_IMPLEMENTATION
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_image(const MnistImage *img)
{
        for (int r = 0; r < MNIST_ROWS; r++) {
                for (int c = 0; c < MNIST_COLS; c++) {
                        uint8_t p = img->pixels[r * MNIST_COLS + c];
                        putchar(p > 192 ? '@' : p > 128 ? '*' : p > 64 ? ':' : p > 0 ? '.' : ' ');
                }
                putchar('\n');
        }
}

int main(int argc, char **argv)
{
        if (argc != 4) {
                fprintf(stderr, "usage: %s <data-dir> <train|test> <index>\n", argv[0]);
                return 2;
        }
        bool test = strcmp(argv[2], "test") == 0;
        size_t idx = strtoul(argv[3], nullptr, 10);

        Mnist data;
        MnistError e = mnist_load(&data, argv[1]);
        if (e.kind != MNIST_OK) {
                char msg[256];
                mnist_error_format(msg, sizeof msg, &e);
                fprintf(stderr, "load failed: %s\n", msg);
                return 1;
        }

        size_t n = test ? data.n_test : data.n_train;
        const MnistImage *images = test ? data.test_images : data.train_images;
        const uint8_t *labels = test ? data.test_labels : data.train_labels;

        if (idx >= n) {
                fprintf(stderr, "index %zu out of range (n=%zu)\n", idx, n);
                mnist_free(&data);
                return 1;
        }

        printf("--- Dataset: %s | Image #%zu | Label: %u ---\n", argv[2], idx, labels[idx]);
        print_image(&images[idx]);
        mnist_free(&data);
}
