/* Optional CPU kernels for Heliax.
 *
 * This file is intentionally small and dependency-free.  It is compiled by
 * scripts/build_native.py for local benchmarking; the pure NumPy backend is
 * always available as a fallback.
 */
#include <math.h>
#include <stddef.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static inline float hx_gelu_tanh(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

void hx_add_relu(const float *a, const float *b, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = fmaxf(a[i] + b[i], 0.0f);
    }
}

void hx_gelu(const float *x, float *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = hx_gelu_tanh(x[i]);
    }
}

void hx_softmax_lastdim(const float *x, float *out, size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; ++row) {
        const float *src = x + row * cols;
        float *dst = out + row * cols;
        float maximum = src[0];
        for (size_t col = 1; col < cols; ++col) {
            if (src[col] > maximum) maximum = src[col];
        }
        float total = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = expf(src[col] - maximum);
            dst[col] = value;
            total += value;
        }
        const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
        for (size_t col = 0; col < cols; ++col) dst[col] *= inverse;
    }
}

void hx_adamw(float *parameter, const float *gradient, float *first_moment,
              float *second_moment, size_t n, float learning_rate, float beta1,
              float beta2, float epsilon, float weight_decay, float bias1,
              float bias2) {
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    for (size_t i = 0; i < n; ++i) {
        first_moment[i] = beta1 * first_moment[i] + one_minus_beta1 * gradient[i];
        second_moment[i] = beta2 * second_moment[i] + one_minus_beta2 * gradient[i] * gradient[i];
        const float first = first_moment[i] / bias1;
        const float second = second_moment[i] / bias2;
        parameter[i] -= learning_rate * (first / (sqrtf(second) + epsilon) + weight_decay * parameter[i]);
    }
}

void hx_layernorm_lastdim(const float *x, const float *gamma, const float *beta,
                          float *out, size_t rows, size_t cols, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        const float *src = x + row * cols;
        float *dst = out + row * cols;
        float mean = 0.0f;
        for (size_t col = 0; col < cols; ++col) mean += src[col];
        mean /= (float)cols;
        float variance = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float delta = src[col] - mean;
            variance += delta * delta;
        }
        variance /= (float)cols;
        const float inverse = 1.0f / sqrtf(variance + eps);
        for (size_t col = 0; col < cols; ++col) {
            dst[col] = (src[col] - mean) * inverse * gamma[col] + beta[col];
        }
    }
}
