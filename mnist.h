// MNIST IDX loader - single-header library (C23)
//
// Reads the four uncompressed IDX files of MNIST (or Fashion-MNIST) from a directory:
// train-images-idx3-ubyte  train-labels-idx1-ubyte  t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte
//
// In exactly one source file, `#define MNIST_IMPLEMENTATION` before including this header.
//
// Mnist m;
// MnistError e = mnist_load(&m, "MNIST");
// if (e.kind != MNIST_OK) { char msg[256]; mnist_error_format(msg, sizeof msg, &e); ... }
// ... m.train_images[0], m.train_labels[0] ...
// mnist_free(&m);

#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>

enum { MNIST_ROWS = 28, MNIST_COLS = 28, MNIST_PIXELS = MNIST_ROWS * MNIST_COLS, MNIST_NUM_LABELS = 10 };

typedef struct {
        uint8_t pixels[MNIST_PIXELS]; // row-major
} MnistImage;

typedef enum {
        MNIST_OK,
        MNIST_ERR_IO,             // file missing, unreadable, ...
        MNIST_ERR_MAGIC,          // wrong magic number: corrupt or wrong file type
        MNIST_ERR_DIMS,           // header image dimensions are not 28x28
        MNIST_ERR_TRUNCATED,      // file shorter than its header says (sizes in bytes)
        MNIST_ERR_COUNT_MISMATCH, // image and label files of a split differ in length
        MNIST_ERR_NOMEM,
} MnistErrorKind;

typedef struct {
        MnistErrorKind kind;
        union {
                struct {
                        uint32_t expected, found;
                } magic;
                struct {
                        uint32_t expected_w, expected_h, found_w, found_h;
                } dims;
                struct {
                        int64_t expected, found;
                } truncated;
                struct {
                        int64_t images, labels;
                } count;
        };
} MnistError;

typedef struct {
        size_t n_train, n_test;
        MnistImage *train_images;
        uint8_t *train_labels;
        MnistImage *test_images;
        uint8_t *test_labels;
} Mnist;

// Loads the four files from `dir`. On error `m` is left empty (nothing to free).
[[nodiscard]] MnistError mnist_load(Mnist *m, const char *dir);

// `mnist_load`, then `mnist_augment_train_with_shifts`.
[[nodiscard]] MnistError mnist_load_with_shift_augmentation(Mnist *m, const char *dir, int max_shift);

// Replaces the training set by every shift variant (-max_shift..=max_shift on both axes,
// including the original) of each training image; each image is followed by its variants,
// all carrying its label. The size grows by (2 * max_shift + 1)^2. The test set is left
// alone: augmenting evaluation data would leak correlated variants across the train/test
// boundary. `max_shift` must be >= 0. On error `m` is unchanged.
[[nodiscard]] MnistError mnist_augment_train_with_shifts(Mnist *m, int max_shift);

// The image shifted by (shift_x, shift_y), i.e. result(x, y) = img(x + shift_x, y + shift_y):
// a positive shift_x moves the content LEFT and a positive shift_y moves it UP. Pixels shifted
// out of frame are dropped and vacated pixels are 0 (black).
MnistImage mnist_shifted(const MnistImage *img, int shift_x, int shift_y);

void mnist_free(Mnist *m);

// snprintf-style: writes a message for `e` into buf, returns the length it needs.
int mnist_error_format(char *buf, size_t cap, const MnistError *e);

#endif // MNIST_H

#ifdef MNIST_IMPLEMENTATION

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_LABELS_MAGIC 2049u
#define MNIST_IMAGES_MAGIC 2051u

static uint32_t mnist_be32_(const uint8_t *b)
{
        return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
}

// Reads a whole IDX file and validates its first `nwords` big-endian u32 header words
// (magic first) into `w`. On success the caller owns *data.
static MnistError mnist_read_idx_(const char *dir, const char *name, uint32_t magic, int nwords,
                                  uint32_t w[], uint8_t **data, int64_t *len)
{
        char path[1024];
        if (snprintf(path, sizeof path, "%s/%s", dir, name) >= (int)sizeof path)
                return (MnistError) {
                .kind = MNIST_ERR_IO
        };
        FILE *f = fopen(path, "rb");
        if (!f)
                return (MnistError) {
                .kind = MNIST_ERR_IO
        };
        long n = fseek(f, 0, SEEK_END) == 0 ? ftell(f) : -1;
        rewind(f);
        uint8_t *d = n < 0 ? nullptr : malloc(n ? (size_t)n : 1);
        size_t got = d ? fread(d, 1, (size_t)n, f) : 0;
        fclose(f);
        if (n < 0 || (d && got != (size_t)n)) {
                free(d);
                return (MnistError) {
                        .kind = MNIST_ERR_IO
                };
        }
        if (!d)
                return (MnistError) {
                .kind = MNIST_ERR_NOMEM
        };

        MnistError e = {.kind = MNIST_OK};
        if (n < 4)
                e = (MnistError) {
                .kind = MNIST_ERR_TRUNCATED, .truncated = {4, n}
        };
        else if ((w[0] = mnist_be32_(d)) != magic)
                e = (MnistError) {
                .kind = MNIST_ERR_MAGIC, .magic = {magic, w[0]}
        };
        else if (n < 4 * nwords)
                e = (MnistError) {
                .kind = MNIST_ERR_TRUNCATED, .truncated = {4 * nwords, n}
        };
        if (e.kind != MNIST_OK) {
                free(d);
                return e;
        }
        for (int i = 1; i < nwords; i++)
                w[i] = mnist_be32_(d + 4 * i);
        *data = d;
        *len = n;
        return e;
}

// Labels of an IDX1 file, one byte each (the file buffer becomes the array).
static MnistError mnist_read_labels_(const char *dir, const char *name, uint8_t **out, size_t *n)
{
        uint32_t w[2];
        uint8_t *d;
        int64_t len;
        MnistError e = mnist_read_idx_(dir, name, MNIST_LABELS_MAGIC, 2, w, &d, &len);
        if (e.kind != MNIST_OK)
                return e;
        if (len < 8 + (int64_t)w[1]) {
                free(d);
                return (MnistError) {
                        .kind = MNIST_ERR_TRUNCATED, .truncated = {8 + (int64_t)w[1], len}
                };
        }
        memmove(d, d + 8, w[1]);
        *out = d;
        *n = w[1];
        return e;
}

// Images of an IDX3 file, 28x28 pixels each (the file buffer becomes the array).
static MnistError mnist_read_images_(const char *dir, const char *name, MnistImage **out, size_t *n)
{
        uint32_t w[4];
        uint8_t *d;
        int64_t len;
        MnistError e = mnist_read_idx_(dir, name, MNIST_IMAGES_MAGIC, 4, w, &d, &len);
        if (e.kind != MNIST_OK)
                return e;
        int64_t need = 16 + (int64_t)w[1] * MNIST_PIXELS;
        if (w[2] != MNIST_ROWS || w[3] != MNIST_COLS)
                e = (MnistError) {
                .kind = MNIST_ERR_DIMS, .dims = {MNIST_COLS, MNIST_ROWS, w[3], w[2]}
        };
        else if (len < need)
                e = (MnistError) {
                .kind = MNIST_ERR_TRUNCATED, .truncated = {need, len}
        };
        if (e.kind != MNIST_OK) {
                free(d);
                return e;
        }
        memmove(d, d + 16, (size_t)w[1] * MNIST_PIXELS);
        *out = (MnistImage *)d;
        *n = w[1];
        return e;
}

MnistError mnist_load(Mnist *m, const char *dir)
{
        size_t n_train_lbl, n_test_lbl;
        *m = (Mnist) {
                0
        };

        MnistError e = mnist_read_labels_(dir, "train-labels-idx1-ubyte", &m->train_labels, &n_train_lbl);
        if (e.kind == MNIST_OK)
                e = mnist_read_images_(dir, "train-images-idx3-ubyte", &m->train_images, &m->n_train);
        if (e.kind == MNIST_OK)
                e = mnist_read_labels_(dir, "t10k-labels-idx1-ubyte", &m->test_labels, &n_test_lbl);
        if (e.kind == MNIST_OK)
                e = mnist_read_images_(dir, "t10k-images-idx3-ubyte", &m->test_images, &m->n_test);
        if (e.kind == MNIST_OK && m->n_train != n_train_lbl)
                e = (MnistError) {
                .kind = MNIST_ERR_COUNT_MISMATCH, .count = {(int64_t)m->n_train, (int64_t)n_train_lbl}
        };
        if (e.kind == MNIST_OK && m->n_test != n_test_lbl)
                e = (MnistError) {
                .kind = MNIST_ERR_COUNT_MISMATCH, .count = {(int64_t)m->n_test, (int64_t)n_test_lbl}
        };
        if (e.kind != MNIST_OK)
                mnist_free(m);
        return e;
}

MnistImage mnist_shifted(const MnistImage *img, int shift_x, int shift_y)
{
        MnistImage out = {0};
        int x0 = shift_x < 0 ? -shift_x : 0;
        int x1 = shift_x > 0 ? MNIST_COLS - shift_x : MNIST_COLS;
        for (int y = 0; y < MNIST_ROWS && x1 > x0; y++) {
                int ys = y + shift_y;
                if (ys >= 0 && ys < MNIST_ROWS)
                        memcpy(&out.pixels[y * MNIST_COLS + x0], &img->pixels[ys * MNIST_COLS + x0 + shift_x],
                               (size_t)(x1 - x0));
        }
        return out;
}

MnistError mnist_augment_train_with_shifts(Mnist *m, int max_shift)
{
        assert(max_shift >= 0);
        size_t side = 2 * (size_t)max_shift + 1;
        if (side > SIZE_MAX / side || m->n_train > SIZE_MAX / sizeof(MnistImage) / (side * side))
                return (MnistError) {
                .kind = MNIST_ERR_NOMEM
        };
        size_t n = m->n_train * side * side;

        MnistImage *images = malloc(n ? n * sizeof *images : 1);
        uint8_t *labels = malloc(n ? n : 1);
        if (!images || !labels) {
                free(images);
                free(labels);
                return (MnistError) {
                        .kind = MNIST_ERR_NOMEM
                };
        }
        size_t at = 0;
        for (size_t i = 0; i < m->n_train; i++)
                for (int sy = -max_shift; sy <= max_shift; sy++)
                        for (int sx = -max_shift; sx <= max_shift; sx++, at++) {
                                images[at] = mnist_shifted(&m->train_images[i], sx, sy);
                                labels[at] = m->train_labels[i];
                        }

        free(m->train_images);
        free(m->train_labels);
        m->train_images = images;
        m->train_labels = labels;
        m->n_train = n;
        return (MnistError) {
                .kind = MNIST_OK
        };
}

MnistError mnist_load_with_shift_augmentation(Mnist *m, const char *dir, int max_shift)
{
        MnistError e = mnist_load(m, dir);
        if (e.kind == MNIST_OK)
                e = mnist_augment_train_with_shifts(m, max_shift);
        if (e.kind != MNIST_OK)
                mnist_free(m);
        return e;
}

void mnist_free(Mnist *m)
{
        free(m->train_images);
        free(m->train_labels);
        free(m->test_images);
        free(m->test_labels);
        *m = (Mnist) {
                0
        };
}

int mnist_error_format(char *buf, size_t cap, const MnistError *e)
{
        switch (e->kind) {
        case MNIST_OK:
                return snprintf(buf, cap, "ok");
        case MNIST_ERR_IO:
                return snprintf(buf, cap, "I/O error: cannot read file");
        case MNIST_ERR_NOMEM:
                return snprintf(buf, cap, "out of memory");
        case MNIST_ERR_MAGIC:
                return snprintf(buf, cap, "Invalid magic number. Expected %u, but found %u",
                                (unsigned)e->magic.expected, (unsigned)e->magic.found);
        case MNIST_ERR_DIMS:
                return snprintf(buf, cap, "Invalid image dimensions. Expected %ux%u, but found %ux%u",
                                (unsigned)e->dims.expected_w, (unsigned)e->dims.expected_h,
                                (unsigned)e->dims.found_w, (unsigned)e->dims.found_h);
        case MNIST_ERR_TRUNCATED:
                return snprintf(buf, cap, "Truncated file. Expected at least %lld bytes, but found %lld",
                                (long long)e->truncated.expected, (long long)e->truncated.found);
        case MNIST_ERR_COUNT_MISMATCH:
                return snprintf(buf, cap, "Count mismatch: %lld images but %lld labels",
                                (long long)e->count.images, (long long)e->count.labels);
        }
        return snprintf(buf, cap, "unknown error");
}

#endif // MNIST_IMPLEMENTATION
