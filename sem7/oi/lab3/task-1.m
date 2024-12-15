function result = f(r)
	R_1_5 = 10*r.^5 - 12*r.^3 + 3*r;
	% Z_1_5 = R_1_5 .* exp(-1i * phi);
	result = R_1_5;
  end
  
R = 5
n = 100

% Создание сетки значений r
r = linspace(0, R, n);

% Вычисление значений функции
f_values = f(r);

f_values2 = zeros(2*n - 1);

for i = 1:(2*n - 1)
    for j = 1:(2*n - 1)
        index = int64(sqrt((j - n)^2 + (i - n)^2)) + 1;
        if index <= n
            f_values2(i, j) = f_values(index);
        else
            f_values2(i, j) = 0; % или другое значение по умолчанию
        end
    end
end

% Вычисление модуля и фазы
magnitude = abs(f_values2);
phase = angle(f_values2);

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