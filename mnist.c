#include "mnist.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define MAGIC_IMAGES 0x00000803u
#define MAGIC_LABELS 0x00000801u

static bool read_be32(FILE *f, uint32_t *out)
{
        uint8_t b[4];
        if (fread(b, 1, 4, f) != 4)
                return false;
        *out = (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
        return true;
}

MnistStatus mnist_load(MnistSet *set, const char *images_path, const char *labels_path)
{
        *set = (MnistSet) {
                0
        };
        MnistStatus st = MNIST_OK;
        uint32_t magic, n_img, n_lbl, rows, cols;

        FILE *fi = fopen(images_path, "rb");
        FILE *fl = fopen(labels_path, "rb");
        if (!fi || !fl) {
                st = MNIST_ERR_IO;
                goto done;
        }

        if (!read_be32(fi, &magic) || !read_be32(fi, &n_img) || !read_be32(fi, &rows) ||
                        !read_be32(fi, &cols)) {
                st = MNIST_ERR_IO;
                goto done;
        }
        if (magic != MAGIC_IMAGES) {
                st = MNIST_ERR_MAGIC;
                goto done;
        }
        if (rows != MNIST_ROWS || cols != MNIST_COLS) {
                st = MNIST_ERR_DIMS;
                goto done;
        }

        if (!read_be32(fl, &magic) || !read_be32(fl, &n_lbl)) {
                st = MNIST_ERR_IO;
                goto done;
        }
        if (magic != MAGIC_LABELS) {
                st = MNIST_ERR_MAGIC;
                goto done;
        }
        if (n_img != n_lbl) {
                st = MNIST_ERR_COUNT_MISMATCH;
                goto done;
        }

        set->n = n_img;
        set->pixels = malloc(set->n * MNIST_PIXELS);
        set->labels = malloc(set->n);
        if (!set->pixels || !set->labels) {
                st = MNIST_ERR_NOMEM;
                goto done;
        }

        if (fread(set->pixels, MNIST_PIXELS, set->n, fi) != set->n ||
                        fread(set->labels, 1, set->n, fl) != set->n)
                st = MNIST_ERR_IO;

done:
        if (fi)
                fclose(fi);
        if (fl)
                fclose(fl);
        if (st != MNIST_OK)
                mnist_free(set);
        return st;
}

void mnist_free(MnistSet *set)
{
        free(set->pixels);
        free(set->labels);
        *set = (MnistSet) {
                0
        };
}

const char *mnist_strerror(MnistStatus st)
{
        switch (st) {
        case MNIST_OK:
                return "ok";
        case MNIST_ERR_IO:
                return "I/O error or truncated file";
        case MNIST_ERR_MAGIC:
                return "bad magic number";
        case MNIST_ERR_DIMS:
                return "unexpected image dimensions";
        case MNIST_ERR_COUNT_MISMATCH:
                return "image/label count mismatch";
        case MNIST_ERR_NOMEM:
                return "out of memory";
        }
        return "unknown error";
}
