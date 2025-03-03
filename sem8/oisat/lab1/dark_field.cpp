#include <iostream>
#include <complex>
#include <vector>
#include <fftw3.h>
#include <matplot/matplot.h>

using namespace std;
using namespace matplot;

int main() {
    const int N = 100;
    const int M = 256;
    const double a = 5.0;
    const double A = 8.0;
    const double B = 0.8;
    const double C = 8.0;
    const double b = (N * N) / (4 * a * M);
    const double h_x = 2 * a / N;
    const double h_u = 2 * b / N;

    vector<double> x(N), y(N);
    for (int i = 0; i < N; ++i) {
        x[i] = -a + i * h_x;
        y[i] = -a + i * h_x;
    }

    vector<complex<double>> w((M - N) / 2, f(N * N), F2(N * N);
    vector<vector<complex<double>>> f_matrix(N, vector<complex<double>>(N));

    // Заполнение матрицы f
    for (int k = 0; k < N; ++k) {
        for (int m = 0; m < N; ++m) {
            f_matrix[k][m] = exp(complex<double>(0, B * M_PI * sin(A * x[k]) * sin(C * y[m])));
        }
    }

    // Преобразование Фурье по строкам
    for (int k = 0; k < N; ++k) {
        vector<complex<double>> row = f_matrix[k];
        vector<complex<double>> padded_row(w.size() + row.size() + w.size());
        copy(w.begin(), w.end(), padded_row.begin());
        copy(row.begin(), row.end(), padded_row.begin() + w.size());
        copy(w.begin(), w.end(), padded_row.begin() + w.size() + row.size());

        fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_row.size());
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_row.size());

        for (size_t i = 0; i < padded_row.size(); ++i) {
            in[i][0] = padded_row[i].real();
            in[i][1] = padded_row[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(padded_row.size(), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        vector<complex<double>> F(padded_row.size());
        for (size_t i = 0; i < padded_row.size(); ++i) {
            F[i] = complex<double>(out[i][0], out[i][1]) * h_x;
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        // Выбор центральной части
        vector<complex<double>> F_shifted(F.begin() + (M - N) / 2, F.begin() + (M - N) / 2 + N);
        f_matrix[k] = F_shifted;
    }

    // Преобразование Фурье по столбцам
    for (int k = 0; k < N; ++k) {
        vector<complex<double>> column(N);
        for (int m = 0; m < N; ++m) {
            column[m] = f_matrix[m][k];
        }

        vector<complex<double>> padded_column(w.size() + column.size() + w.size());
        copy(w.begin(), w.end(), padded_column.begin());
        copy(column.begin(), column.end(), padded_column.begin() + w.size());
        copy(w.begin(), w.end(), padded_column.begin() + w.size() + column.size());

        fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_column.size());
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_column.size());

        for (size_t i = 0; i < padded_column.size(); ++i) {
            in[i][0] = padded_column[i].real();
            in[i][1] = padded_column[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(padded_column.size(), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        vector<complex<double>> F(padded_column.size());
        for (size_t i = 0; i < padded_column.size(); ++i) {
            F[i] = complex<double>(out[i][0], out[i][1]) * h_x;
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        // Выбор центральной части
        vector<complex<double>> F_shifted(F.begin() + (M - N) / 2, F.begin() + (M - N) / 2 + N);
        for (int m = 0; m < N; ++m) {
            f_matrix[m][k] = F_shifted[m];
        }
    }

    // Создание маски d
    vector<vector<double>> d(N, vector<double>(N, 1.0));
    double x1 = -a;
    double y1 = -a;
    for (int k = 0; k < N; ++k) {
        if (x1 >= -2 && x1 <= 2) {
            y1 = -a;
            for (int m = 0; m < N; ++m) {
                if (y1 >= -2 && y1 <= 2) {
                    d[k][m] = 0.0;
                }
                y1 += h_x;
            }
        }
        x1 += h_x;
    }

    // Применение маски
    for (int k = 0; k < N; ++k) {
        for (int m = 0; m < N; ++m) {
            F2[k * N + m] = f_matrix[k][m] * d[k][m];
        }
    }

    // Обратное преобразование Фурье по строкам
    for (int k = 0; k < N; ++k) {
        vector<complex<double>> row(N);
        for (int m = 0; m < N; ++m) {
            row[m] = F2[k * N + m];
        }

        vector<complex<double>> padded_row(w.size() + row.size() + w.size());
        copy(w.begin(), w.end(), padded_row.begin());
        copy(row.begin(), row.end(), padded_row.begin() + w.size());
        copy(w.begin(), w.end(), padded_row.begin() + w.size() + row.size());

        fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_row.size());
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_row.size());

        for (size_t i = 0; i < padded_row.size(); ++i) {
            in[i][0] = padded_row[i].real();
            in[i][1] = padded_row[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(padded_row.size(), in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        vector<complex<double>> F(padded_row.size());
        for (size_t i = 0; i < padded_row.size(); ++i) {
            F[i] = complex<double>(out[i][0], out[i][1]) * h_x;
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        // Выбор центральной части
        vector<complex<double>> F_shifted(F.begin() + (M - N) / 2, F.begin() + (M - N) / 2 + N);
        for (int m = 0; m < N; ++m) {
            F2[k * N + m] = F_shifted[m];
        }
    }

    // Обратное преобразование Фурье по столбцам
    for (int k = 0; k < N; ++k) {
        vector<complex<double>> column(N);
        for (int m = 0; m < N; ++m) {
            column[m] = F2[m * N + k];
        }

        vector<complex<double>> padded_column(w.size() + column.size() + w.size());
        copy(w.begin(), w.end(), padded_column.begin());
        copy(column.begin(), column.end(), padded_column.begin() + w.size());
        copy(w.begin(), w.end(), padded_column.begin() + w.size() + column.size());

        fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_column.size());
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * padded_column.size());

        for (size_t i = 0; i < padded_column.size(); ++i) {
            in[i][0] = padded_column[i].real();
            in[i][1] = padded_column[i].imag();
        }

        fftw_plan p = fftw_plan_dft_1d(padded_column.size(), in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        vector<complex<double>> F(padded_column.size());
        for (size_t i = 0; i < padded_column.size(); ++i) {
            F[i] = complex<double>(out[i][0], out[i][1]) * h_x;
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        // Выбор центральной части
        vector<complex<double>> F_shifted(F.begin() + (M - N) / 2, F.begin() + (M - N) / 2 + N);
        for (int m = 0; m < N; ++m) {
            F2[m * N + k] = F_shifted[m];
        }
    }

    // Визуализация результатов
    vector<vector<double>> abs_F2(N, vector<double>(N));
    for (int k = 0; k < N; ++k) {
        for (int m = 0; m < N; ++m) {
            abs_F2[k][m] = abs(F2[k * N + m]);
        }
    }

    imagesc(abs_F2);
    colorbar();
    show();

    return 0;
}