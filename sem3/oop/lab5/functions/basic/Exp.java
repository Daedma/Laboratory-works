package functions.basic;

import functions.Function;

public class Exp implements Function {

	public double getRightDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	public double getLeftDomainBorder() {
		return Double.NEGATIVE_INFINITY;
	}

	public double getFunctionValue(double x) {
		return Math.exp(x);
	}
}
