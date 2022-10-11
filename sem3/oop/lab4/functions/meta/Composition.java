package functions.meta;

import functions.Function;

public class Composition implements Function {
	Function _external, _internal;

	public Composition(Function external, Function internal) {
		_external = external;
		_internal = internal;
	}

	public double getLeftDomainBorder() {
		return _external.getLeftDomainBorder();
	}

	public double getRightDomainBorder() {
		return _external.getRightDomainBorder();
	}

	public double getFunctionValue(double x) {
		return _external.getFunctionValue(_internal.getFunctionValue(x));
	}
}
