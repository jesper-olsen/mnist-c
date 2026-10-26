#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_image(const uint8_t *img)
{
        for (int r = 0; r < MNIST_ROWS; r++) {
                for (int c = 0; c < MNIST_COLS; c++) {
                        uint8_t p = img[r * MNIST_COLS + c];
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
        const char *dir = argv[1];
        const char *prefix = strcmp(argv[2], "test") == 0 ? "t10k" : "train";
        size_t idx = strtoul(argv[3], nullptr, 10);

        char ipath[512], lpath[512];
        snprintf(ipath, sizeof ipath, "%s/%s-images-idx3-ubyte", dir, prefix);
        snprintf(lpath, sizeof lpath, "%s/%s-labels-idx1-ubyte", dir, prefix);

        MnistSet set;
        MnistStatus st = mnist_load(&set, ipath, lpath);
        if (st != MNIST_OK) {
                fprintf(stderr, "load failed: %s\n", mnist_strerror(st));
                return 1;
        }
        if (idx >= set.n) {
                fprintf(stderr, "index %zu out of range (n=%zu)\n", idx, set.n);
                mnist_free(&set);
                return 1;
        }

        printf("--- Dataset: %s | Image #%zu | Label: %u ---\n", argv[2], idx, set.labels[idx]);
        print_image(mnist_image(&set, idx));
        mnist_free(&set);
}
