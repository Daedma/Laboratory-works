% Параметры гауссова пучка
sigma = 1; % Ширина гауссова пучка
N = 1024; % Количество точек
x = linspace(-10, 10, N); % Диапазон x
u = linspace(-10, 10, N); % Диапазон u

% Гауссов пучок
f = exp(-x.^2 / (2 * sigma^2));

% Финитное преобразование Фурье методом прямоугольников
F = zeros(size(u));
dx = x(2) - x(1);

for k = 1:length(u)
    integrand = f .* exp(-1i * u(k) * x);
    F(k) = sum(integrand) * dx;
end

% Амплитуда и фаза
amplitude = abs(F);
phase = angle(F);

% Построение графиков
figure;
subplot(2, 1, 1);
plot(u, amplitude);
title('Амплитуда');
xlabel('u');
ylabel('Амплитуда');

subplot(2, 1, 2);
plot(u, phase);
title('Фаза');
xlabel('u');
ylabel('Фаза');
