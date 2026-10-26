#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>

enum { MNIST_ROWS = 28, MNIST_COLS = 28, MNIST_PIXELS = MNIST_ROWS * MNIST_COLS };

typedef enum {
        MNIST_OK,
        MNIST_ERR_IO,
        MNIST_ERR_MAGIC,
        MNIST_ERR_DIMS,
        MNIST_ERR_COUNT_MISMATCH,
        MNIST_ERR_NOMEM,
} MnistStatus;

typedef struct {
        size_t n;
        uint8_t *pixels; // n * MNIST_PIXELS, row-major
        uint8_t *labels; // n
} MnistSet;

[[nodiscard]] MnistStatus mnist_load(MnistSet *set, const char *images_path,
                                     const char *labels_path);
void mnist_free(MnistSet *set);
const char *mnist_strerror(MnistStatus st);

static inline const uint8_t *mnist_image(const MnistSet *s, size_t i)
{
        return s->pixels + i * MNIST_PIXELS;
}

#endif
