% Определение диапазона значений x
x = linspace(-5, 5, 1000);

% Определение гауссова пучка
gaussian_beam = exp(-x.^2);

% Разделение на амплитуду и фазу
amplitude = abs(gaussian_beam);
phase = angle(gaussian_beam);

% Построение графика амплитуды
figure;
subplot(2, 1, 1);
plot(x, amplitude, 'LineWidth', 2);
title('Амплитуда гауссова пучка');
xlabel('x');
ylabel('Амплитуда');
grid on;

% Построение графика фазы
subplot(2, 1, 2);
plot(x, phase, 'LineWidth', 2);
title('Фаза гауссова пучка');
xlabel('x');
ylabel('Фаза');
grid on;
