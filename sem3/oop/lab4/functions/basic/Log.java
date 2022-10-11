package functions.basic;

import functions.Function;

public class Log implements Function {

	private double _base;

	public Log(double base) {
		if (base <= 0 || base == 1)
			throw IllegalArgumentException(
					"The base of a logarithm cannot be <= 0 or equal 1 (base: "
							+ base + ")");
		_base = base;
	}

	public Log() {
		_base = Math.E;
	}

	public double getRightDomainBorder() {
		return 0.;
	}

	public double getLeftDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	public double getFunctionValue(double x) {
		return Math.log(x) / Math.log(_base);
	}
}
