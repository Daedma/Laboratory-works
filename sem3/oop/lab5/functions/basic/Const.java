package functions.basic;

import functions.Function;

public class Const implements Function {
	private double value;

	public Const(double c) {
		value = c;
	}

	public double getLeftDomainBorder() {
		return Double.NEGATIVE_INFINITY;
	}

	public double getRightDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	public double getFunctionValue(double c) {
		return value;
	}
}
