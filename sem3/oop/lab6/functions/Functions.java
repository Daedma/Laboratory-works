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
