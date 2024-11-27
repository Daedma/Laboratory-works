
% Для аналитики применить не финитное
% преобразование. Использовать свойства
% преобразования Фурье.
% Параметры
N = 500; % Размерность вектора f
M = 2^12; % Размерность после дополнения нулями
a = 3;
hx = 2 * a / N; % Шаг по x
input_field = @(x) sin(4*pi*x);

% Шаг 1: Дискретизация входной функции f(x)
x = linspace(-a, a, N);
f = input_field(x);

% Шаг 2: Дополнение вектора f нулями до размерности M
f_padded = [zeros(1, (M-N)/2), f, zeros(1, (M-N)/2)];

% Шаг 3: Разбить вектор f на две половины и поменять их местами
f_swapped = [f_padded((M/2+1):end), f_padded(1:(M/2))];

% Шаг 4: Выполнить БПФ от f и умножить результат на шаг hx
F = fft(f_swapped) * hx;

% Шаг 5: Разбить вектор F на две половины и поменять их местами
F = [F((M/2+1):end), F(1:(M/2))];

% Шаг 6: «Вырезать» центральную часть вектора F, оставив центральные N элементов
F = F((M/2-N/2+1):(M/2+N/2));

% Шаг 7: Пересчитать область задания функции Fa(u) по формуле (7)
b = N^2 / (4 * a * M);
u = linspace(-b, b, N);

analytical_F = -a*1i*sinc(2*(2-u)*a) + a*1i*sinc(2*(2+u)*a);

% Вывод результатов
figure;
subplot(2,2,1);
plot(x, abs(f));
title('Амплитуда f(x)');
xlabel('x');
ylabel('|f(x)|');

subplot(2,2,3);
plot(x, angle(f));
title('Фаза f(x)');
xlabel('x');
ylabel('∠f(x)');

subplot(2,2,2);
plot(u, abs(F));
hold on;
plot(u, abs(analytical_F));
hold off;
title('Амплитуда F(u)');
xlabel('u');
ylabel('|F(u)|');
legend('Numerical', 'Analitical');

subplot(2,2,4);
plot(u, angle(F));
hold on;
plot(u, angle(analytical_F));
hold off;
title('Фаза F(u)');
xlabel('u');
ylabel('∠F(u)');
legend('Numerical', 'Analitical');