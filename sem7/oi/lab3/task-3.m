m = 5  
R = 5
n = 4096

function result = f(r)
	R_1_5 = 10*r.^5 - 12*r.^3 + 3*r;
	result = R_1_5;
  end

r = linspace(0, R, n); % Диапазон r
rho = linspace(0, R, n); % Диапазон rho


% Финитное преобразование Фурье методом прямоугольников
F = zeros(size(rho));
dr = r(2) - r(1);

tic

f_values = f(r);

for k = 1:n
	integrand = f_values .* besselj(m, 2 * pi * rho(k) * r) .* r;
	F(k) = 2 * pi * 1i^(-m) * sum(integrand) * dr;
end

toc

a = zeros(2*n - 1);

b = zeros(2*n - 1);

for j = 1:(2*n - 1)
    for k = 1:(2*n - 1)
        alpha = int64(sqrt((j - n)^2 + (k - n)^2)) + 1;
        if alpha <= n
            a(j, k) = F(alpha);
            b(j, k) = F(alpha) * exp(1i * m * atan2(k - n, j - n));
        end
    end
end


Вычисление модуля и фазы
magnitude = abs(a);
phase = angle(a);

% Построение графиков
figure;
subplot(2, 2, 1);
imagesc(magnitude);
title('Модуль 𝐹(𝜌)');
colorbar;

subplot(2, 2, 2);
imagesc(phase);
title('Фаза 𝐹(𝜌)');
colorbar;

% Вычисление модуля и фазы
magnitude = abs(b);
phase = angle(b);

% Построение графиков
subplot(2, 2, 3);
imagesc(magnitude);
title('Модуль 𝐹(𝜌) exp(𝑖𝑚𝜃)');
colorbar;

subplot(2, 2, 4);
imagesc(phase);
title('Фаза  𝐹(𝜌) exp(𝑖𝑚𝜃)');
colorbar;