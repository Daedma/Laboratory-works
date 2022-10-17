package functions.basic;

import functions.Function;

public class Log implements Function {

	private double _base;

	public Log(double base) throws IllegalArgumentException {
		if (base <= 0 || base == 1)
			throw new IllegalArgumentException(
					"The base of a logarithm cannot be <= 0 or equal 1 (base: "
							+ base + ")");
		_base = base;
	}

	public Log() {
		_base = Math.E;
	}

	public double getRightDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	public double getLeftDomainBorder() {
		return 0.;
	}

	public double getFunctionValue(double x) {
		return Math.log(x) / Math.log(_base);
	}
}
