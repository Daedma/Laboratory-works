#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <matplot/matplot.h>

#define M_PI 3.14159265358979323846


int main() {
    // Параметры сигнала
    const int N = 64; // Количество точек
    const double T = 1.0 / 128.0; // Период дискретизации
    std::vector<double> signal(N);

    // Генерация синусоидального сигнала
    for (int i = 0; i < N; ++i) {
        signal[i] = sin(2.0 * M_PI * 10.0 * i * T); // Синусоида с частотой 10 Гц
    }

    // Выделение памяти для FFT
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Заполнение входного массива
    for (int i = 0; i < N; ++i) {
        in[i][0] = signal[i]; // Действительная часть
        in[i][1] = 0.0;      // Мнимая часть
    }

    // Выполнение FFT
    fftw_execute(p);

    // Вычисление амплитудного спектра
    std::vector<double> amplitude(N);
    for (int i = 0; i < N; ++i) {
        amplitude[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    // Визуализация исходного сигнала и его спектра
    using namespace matplot;

    subplot(2, 1, 0);
    plot(signal);
    title("Исходный сигнал");

    subplot(2, 1, 1);
    plot(amplitude);
    title("Амплитудный спектр");

    show();

    // Освобождение памяти
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return 0;
}