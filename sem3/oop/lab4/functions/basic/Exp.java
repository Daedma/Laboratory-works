package functions.basic;

import functions.Function;

public class Exp implements Function {

	double getRightDomainBorder() {
		return Double.POSITIVE_INFINITY;
	}

	double getLeftDomainBorder() {
		return Double.NEGATIVE_INFINITY;
	}

	double getFunctionValue(double x) {
		return Math.exp(x);
	}
}
