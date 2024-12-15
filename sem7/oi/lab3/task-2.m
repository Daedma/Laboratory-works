m = 5  
R = 5
n = 100

function result = f(r)
	R_1_5 = 10*r.^5 - 12*r.^3 + 3*r;
	% Z_1_5 = R_1_5 .* exp(-1i * phi);
	result = R_1_5;
  end

% Создание сетки значений r
r = linspace(0, R, n);

% Вычисление значений функции
f_values = f(r);

a = zeros(2*n - 1);

for j = 1:(2*n - 1)
    for k = 1:(2*n - 1)
        alpha = int64(sqrt((j - n)^2 + (k - n)^2)) + 1;
        if alpha <= n
            a(j, k) = f_values(alpha) * exp(1i * m * atan2(k - n, j - n));
        else
            a(j, k) = 0; % или другое значение по умолчанию
        end
    end
end

% Вычисление модуля и фазы
magnitude = abs(a);
phase = angle(a);

% Построение графиков
figure;
subplot(1, 2, 1);
imagesc(magnitude);
title('Модуль');
colorbar;

subplot(1, 2, 2);
imagesc(phase);
title('Фаза');
colorbar;