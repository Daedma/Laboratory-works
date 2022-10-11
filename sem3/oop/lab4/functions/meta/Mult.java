package functions.meta;

import functions.Function;

public class Mult implements Function {
	Function first, second;

	public Mult(Function rhs, Function lhs) {
		first = rhs;
		second = lhs;
	}

	public double getLeftDomainBorder() {
		double maxLeftDomainBorder = Math.max(first.getLeftDomainBorder(), second.getRightDomainBorder());
		double minRightDomainBorder = Math.min(first.getRightDomainBorder(), second.getRightDomainBorder());
		return minRightDomainBorder < maxLeftDomainBorder ? Double.NaN : maxLeftDomainBorder;
	}

	public double getRightDomainBorder() {
		double maxLeftDomainBorder = Math.max(first.getLeftDomainBorder(), second.getRightDomainBorder());
		double minRightDomainBorder = Math.min(first.getRightDomainBorder(), second.getRightDomainBorder());
		return minRightDomainBorder < maxLeftDomainBorder ? Double.NaN : minRightDomainBorder;
	}

	public double getFunctionValue(double x) {
		return first.getFunctionValue(x) * second.getFunctionValue(x);
	}
}
