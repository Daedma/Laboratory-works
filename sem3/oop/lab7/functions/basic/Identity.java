package functions.basic;

import functions.Function;

public class Identity implements Function {
	public double getLeftDomainBorder() {
		return Double.NEGATIVE_INFINITY;
	}

	public double getRightDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	public double getFunctionValue(double x) {
		return x;
	}
}
