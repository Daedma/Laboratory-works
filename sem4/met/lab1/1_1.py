import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from scipy import optimize

## 8 вариант
# 1
print("1 задание:")
A = np.random.uniform(0, 2, (5, 5))
print(A)
np.transpose(A)
print(np.linalg.det(A))

# 2
print("2 задание: ")
B = np.array([1, 2 ,3, 4, 5], int)
C = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16], [17, 18, 19, 20, 21]], int)
print("B*A = ", B*A)
print("A*C = ", A*C)

# 3
print("3 задание: ")
A = np.array([[2, 1, -3], [0, 1, -1], [0, -2, 2]], int)
np.linalg.eig(A)

# 4
print("4 задание: ")
print(integrate.quad(lambda x: (1/(1+math.sqrt(2*x + 1))), 1, 4))

# 5
print("5 задание: ")
def f(x, y):
    return math.cos(x + y)

def h(x):
    return x;
	
print(integrate.dblquad(f, 0, math.pi/2, 0, h));

# 6
print("6 задание: ")
# Создание объектов артборда и холста

plt.figure(figsize=(8, 5), dpi=80)
ax = plt.subplot(111)

# Мы решили удалить правую и верхнюю прямоугольные границы
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# Установить направление данных на координатной оси
 # 0 согласуется с нашей общей декартовой системой координат, 1 - противоположность
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


# Подготовить данные, использовать распаковку последовательности

X = np.linspace(-20, 2, 256, endpoint=False)
C, L = np.log(2 - X), -X/2

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="ln(2 - x) Function")
plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="-x/2 Function")

# x1, x2 = optimize.root_scalar(lambda x: np.log(2 - x) + x/2)
# y1, y2 = np.log(2 - x1), np.log(2 - x2)

plt.xlim(X.min(), X.max())

# Изменить метку на оси координат 
# plt.xticks([-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2 ],
        #    [r'$-20$', r'$-18$', r'$-16$', r'$-14$', r'$-12$', r'$-10$', r'$-8$', r'$-6$', r'$-4$', r'$-2$', r'$0$', r'$2$' ])

plt.ylim(C.min(), C.max())
# plt.yticks([-2, -1, 1, 2])

def interpolated_intercepts(x, y1, y2):
    """Find the intercepts of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

        R = intersection(L1, L2)

        return R
    
    xcs, ycs = [], []
    for idx in np.argwhere(np.diff(np.sign(y1 - y2)) != 0):
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)

x = np.arange(-20, 2 - 1.e-10, 0.1)

for xc, yc in zip(*interpolated_intercepts(x, np.log(2 - x), -x/2)):
    plt.plot(xc, yc, 'co', ms=5)
    print(xc, yc)

ax.set_xlabel("X", fontsize=15, labelpad=0)
ax.set_ylabel("Y", fontsize=15, labelpad=0, rotation = 0)
plt.legend(loc='upper left', frameon=False)
plt.grid()
plt.show()