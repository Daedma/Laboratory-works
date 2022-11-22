package functions;

public interface TabulatedFunctionFactory {
	TabulatedFunction createTabulatedFunction(double leftX, double rightX, int pointsCount)
			throws IllegalArgumentException;

	TabulatedFunction createTabulatedFunction(double leftX, double rightX, double[] values)
			throws IllegalArgumentException;

	TabulatedFunction createTabulatedFunction(FunctionPoint[] values) throws IllegalArgumentException;
}
