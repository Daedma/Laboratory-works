package functions;

import functions.Function;
import functions.basic.Const;
import functions.meta.Composition;
import functions.meta.Mult;
import functions.meta.Power;
import functions.meta.Scale;
import functions.meta.Shift;
import functions.meta.Sum;

final public class Functions {
	private Functions() {
	}

	public static double integrate(Function f, double left, double right, double step) {
		if (left < f.getLeftDomainBorder() || right > f.getRightDomainBorder()) {
			throw new IllegalArgumentException(
					"The domain of integration is not included in the domain of the function");
		}
		if (left == right) {
			return 0;
		}
		double result = 0;
		double sign = +1;
		if (left > right) {
			sign = -1;
			double tmp = left;
			left = right;
			right = tmp;
		}
		double cur;
		for (cur = left; cur + step <= right; cur += step) {
			result += (f.getFunctionValue(cur) + f.getFunctionValue(cur + step)) / 2 * step;
		}
		result += (f.getFunctionValue(cur) + f.getFunctionValue(right)) / 2 * step;
		return result * sign;
	}

	public static Function shift(Function f, double shiftX, double shiftY) {
		return new Shift(f, shiftX, shiftY);
	}

	public static Function scale(Function f, double scaleX, double scaleY) {
		return new Scale(f, scaleX, scaleY);
	}

	public static Function power(Function f, double power) {
		return new Power(f, new Const(power));
	}

	public static Function power(Function f, Function power) {
		return new Power(f, power);
	}

	public static Function sum(Function f1, Function f2) {
		return new Sum(f1, f2);
	}

	public static Function mult(Function f1, Function f2) {
		return new Mult(f1, f2);
	}

	public static Function composition(Function f1, Function f2) {
		return new Composition(f1, f2);
	}
}
