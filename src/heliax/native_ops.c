/* Optional CPU kernels for Heliax.
 *
 * This file is intentionally small and dependency-free.  It is compiled by
 * scripts/build_native.py for local benchmarking; the pure NumPy backend is
 * always available as a fallback.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static inline float hx_gelu_tanh(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
}

void hx_add_relu(const float *a, const float *b, float *out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        out[i] = fmaxf(a[i] + b[i], 0.0f);
    }
}

void hx_gelu(const float *x, float *out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        out[i] = hx_gelu_tanh(x[i]);
    }
}

void hx_softmax_lastdim(const float *x, float *out, size_t rows, size_t cols) {
    #pragma omp parallel for if (rows > 1)
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

void hx_adagrad(float *parameter, const float *gradient, float *accumulator,
                size_t n, float learning_rate, float epsilon) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        accumulator[i] += gradient[i] * gradient[i];
        parameter[i] -= learning_rate * gradient[i] / (sqrtf(accumulator[i]) + epsilon);
    }
}

void hx_rmsprop(float *parameter, const float *gradient, float *square_average,
                size_t n, float learning_rate, float decay, float epsilon) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        square_average[i] = decay * square_average[i] + (1.0f - decay) * gradient[i] * gradient[i];
        parameter[i] -= learning_rate * gradient[i] / (sqrtf(square_average[i]) + epsilon);
    }
}

void hx_silu(const float *x, float *out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void hx_mae(const float *prediction, const float *target, float *loss,
            float *gradient, size_t n) {
    float total = 0.0f;
    const float inverse_n = n > 0 ? 1.0f / (float)n : 0.0f;
    #pragma omp parallel for reduction(+ : total) if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        total += fabsf(prediction[i] - target[i]);
    }
    *loss = total * inverse_n;
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float difference = prediction[i] - target[i];
        gradient[i] = (difference > 0.0f ? 1.0f : difference < 0.0f ? -1.0f : 0.0f) * inverse_n;
    }
}

void hx_bce_with_logits(const float *logits, const float *target, float *loss,
                        float *gradient, size_t n) {
    float total = 0.0f;
    const float inverse_n = n > 0 ? 1.0f / (float)n : 0.0f;
    #pragma omp parallel for reduction(+ : total) if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float x = logits[i];
        const float t = target[i];
        total += fmaxf(x, 0.0f) - x * t + log1pf(expf(-fabsf(x)));
    }
    *loss = total * inverse_n;
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float x = logits[i];
        gradient[i] = (1.0f / (1.0f + expf(-x)) - target[i]) * inverse_n;
    }
}

void hx_huber(const float *prediction, const float *target, float *loss,
              float *gradient, size_t n, float delta) {
    float total = 0.0f;
    const float inverse_n = n > 0 ? 1.0f / (float)n : 0.0f;
    #pragma omp parallel for reduction(+ : total) if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float difference = prediction[i] - target[i];
        const float absolute = fabsf(difference);
        total += absolute <= delta ? 0.5f * absolute * absolute
                                    : delta * (absolute - 0.5f * delta);
    }
    *loss = total * inverse_n;
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float difference = prediction[i] - target[i];
        const float absolute = fabsf(difference);
        const float derivative = absolute <= delta ? difference
                                                  : (difference < 0.0f ? -delta : delta);
        gradient[i] = derivative * inverse_n;
    }
}

void hx_mse(const float *prediction, const float *target, float *loss,
            float *gradient, size_t n) {
    float squared_error = 0.0f;
    const float inverse_n = n > 0 ? 1.0f / (float)n : 0.0f;
    #pragma omp parallel for reduction(+ : squared_error) if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        const float difference = prediction[i] - target[i];
        squared_error += difference * difference;
    }
    *loss = squared_error * inverse_n;
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i) {
        gradient[i] = 2.0f * (prediction[i] - target[i]) * inverse_n;
    }
}

void hx_cross_entropy_lastdim(const float *logits, const int64_t *targets,
                              float *loss, float *gradient, size_t rows,
                              size_t cols) {
    float loss_sum = 0.0f;
    const float inverse_rows = rows > 0 ? 1.0f / (float)rows : 0.0f;
    #pragma omp parallel for reduction(+ : loss_sum) if (rows > 1)
    for (size_t row = 0; row < rows; ++row) {
        const float *src = logits + row * cols;
        float maximum = src[0];
        for (size_t col = 1; col < cols; ++col) {
            if (src[col] > maximum) maximum = src[col];
        }
        float exponential_sum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            exponential_sum += expf(src[col] - maximum);
        }
        const float log_normalizer = maximum + logf(exponential_sum);
        const size_t target = (size_t)targets[row];
        loss_sum += log_normalizer - src[target];
        float *dst = gradient + row * cols;
        for (size_t col = 0; col < cols; ++col) {
            dst[col] = (expf(src[col] - log_normalizer) - (col == target ? 1.0f : 0.0f)) * inverse_rows;
        }
    }
    *loss = loss_sum * inverse_rows;
}

void hx_log_softmax_lastdim(const float *x, float *out, size_t rows, size_t cols) {
    #pragma omp parallel for if (rows > 1)
    for (size_t row = 0; row < rows; ++row) {
        const float *src = x + row * cols;
        float *dst = out + row * cols;
        float maximum = src[0];
        for (size_t col = 1; col < cols; ++col) {
            if (src[col] > maximum) maximum = src[col];
        }
        float total = 0.0f;
        for (size_t col = 0; col < cols; ++col) total += expf(src[col] - maximum);
        const float log_normalizer = maximum + logf(total);
        for (size_t col = 0; col < cols; ++col) dst[col] = src[col] - log_normalizer;
    }
}

void hx_adamw(float *parameter, const float *gradient, float *first_moment,
              float *second_moment, size_t n, float learning_rate, float beta1,
              float beta2, float epsilon, float weight_decay, float bias1,
              float bias2) {
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    #pragma omp parallel for if (n > 4096)
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
    #pragma omp parallel for if (rows > 1)
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
