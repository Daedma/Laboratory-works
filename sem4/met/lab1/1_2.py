import sympy


print("1 задание: ")
x = sympy.Symbol('x')
y = sympy.Symbol('y')

expr = (2/(x**2) + 3/(y**3))*(x+3*y)

sympy.simplify(expr)
print(float(expr.subs(x, -2.01).subs(y, sympy.sqrt(5))))

print("2 задание: ")
print(sympy.diff(expr, x))
print(sympy.diff(expr, y))

print("3 задание: ")
sympy.integrate()