import sympy

print("1 задание: ")
x = sympy.Symbol('x')
y = sympy.Symbol('y')

expr = (2/(x**2) + 3/(y**3))*(x+3*y)

sympy.simplify(expr)
print(expr.subs(x, -2.01).subs(y, sympy.sqrt(5)))

print("2 задание: ")
print(sympy.diff(expr, x))
print(sympy.diff(expr, y))

print("3 задание: ")
print(sympy.integrate(expr, x))

print("4 задание: ")
print(sympy.solveset(sympy.Eq(expr.subs(x, 2), -100), x))

print("5 задание:")
system = sympy.Matrix([[-7, 3, 1, 0], [2, -1, 1, -1], [1, -2, 5, -2]])
print(sympy.linsolve(system))

print("6 задание: ")
expr = 1/(1 + sympy.sqrt(2*x + 1));
print(sympy.integrate(expr, (x, 0, 4)))

print("7 задание: ")
expr = sympy.cos(x + y)
print(sympy.integrate(expr, (y, 0, x), (x, 0, sympy.pi/2)))