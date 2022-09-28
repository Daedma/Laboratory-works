package functions;

public interface TabulatedFunction {
	double getLeftDomainBorder();

	double getFunctionValue(double x);

	int getPointsCount();

	FunctionPoint getPoint(int index) throws FunctionPointIndexOutOfBoundsException;

	void setPoint(int index, FunctionPoint point)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException;

	double getPointX(int index) throws FunctionPointIndexOutOfBoundsException;

	void setPointX(int index, double x)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException;

	double getPointY(int index) throws FunctionPointIndexOutOfBoundsException;

	void setPointY(int index, double y) throws FunctionPointIndexOutOfBoundsException;

	void deletePoint(int index) throws FunctionPointIndexOutOfBoundsException;

	void addPoint(FunctionPoint point) throws InappropriateFunctionPointException;
}
