% Параметры
n = 1000;
x = linspace(-10, 10, n); % Диапазон значений x

% Исходное значение beta
beta = 1/10;

% Функция f(x) = exp(i * beta * x)
f = @(x, beta) exp(1i * beta * x);

% Построение графиков для различных значений beta
betas = [1/10, 1, 10, -1/10, -1, -10];

figure;
for i = 1:length(betas)
    beta_current = betas(i);
    y = f(x, beta_current);

    % Амплитуда
    subplot(length(betas), 2, 2*i-1);
    plot(x, abs(y));
    title(['Амплитуда для \beta = ', num2str(beta_current)]);
    xlabel('x');
    ylabel('Амплитуда');

    % Фаза
    subplot(length(betas), 2, 2*i);
    plot(x, angle(y));
    title(['Фаза для \beta = ', num2str(beta_current)]);
    xlabel('x');
    ylabel('Фаза');
end

% Возвращение параметра beta в начальное значение
beta = 1/10;