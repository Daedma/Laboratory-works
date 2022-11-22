package functions.meta;

import functions.Function;

public class Power implements Function {
	Function _base, _exp;

	public Power(Function base, Function exp) {
		_base = base;
		_exp = exp;
	}

	public double getLeftDomainBorder() {
		return _base.getLeftDomainBorder();
	}

	public double getRightDomainBorder() {
		return _base.getRightDomainBorder();
	}

	public double getFunctionValue(double x) {
		return Math.pow(_base.getFunctionValue(x), _exp.getFunctionValue(x));
	}
}
