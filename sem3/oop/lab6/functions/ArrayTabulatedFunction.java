package functions;

import java.io.Serializable;

public class ArrayTabulatedFunction implements TabulatedFunction, Serializable {
	// Members
	private FunctionPoint[] _points;
	private int _pointsCount;

	private static final double REALLOC_MULTIPLIER = 2.0;

	// Private methods
	private double getLinearFunctionValue(FunctionPoint firstPoint, FunctionPoint secondPoint, double x) {
		return (x - firstPoint.getX()) * (secondPoint.getY() - firstPoint.getY()) /
				(secondPoint.getX() - firstPoint.getX())
				+ firstPoint.getY();
	}

	private boolean isEmpty() {
		return _pointsCount == 0;
	}

	private boolean isValidIndex(int index) {
		return index >= 0 && index < _pointsCount;
	}

	private boolean isContains(double x) {
		for (int i = 0; i != _pointsCount; ++i)
			if (_points[i].getX() == x)
				return true;
		return false;
	}

	private int getPlaceForPoint(double x) {
		for (int i = _pointsCount - 1; i >= 0; --i) {
			if (x > _points[i].getX())
				return i + 1;
		}
		return 0;
	}

	private void realloc() {
		FunctionPoint[] buff = new FunctionPoint[(int) ((_points.length == 0 ? 1 : _points.length)
				* REALLOC_MULTIPLIER)];
		System.arraycopy(_points, 0, buff, 0, _pointsCount);
		_points = buff;
	}

	// Constructors
	public ArrayTabulatedFunction(double leftX, double rightX, int pointsCount) throws IllegalArgumentException {
		if (leftX >= rightX)
			throw new IllegalArgumentException("leftX >= rightX");
		if (pointsCount < 2)
			throw new IllegalArgumentException("Points count less than 2");
		_pointsCount = pointsCount;
		_points = new FunctionPoint[pointsCount];
		double step = Math.abs(rightX - leftX) / (pointsCount - 1);
		for (int i = 0; i != pointsCount; ++i) {
			_points[i] = new FunctionPoint(leftX, 0);
			leftX += step;
		}
	}

	public ArrayTabulatedFunction(double leftX, double rightX, double[] values) throws IllegalArgumentException {
		if (leftX >= rightX)
			throw new IllegalArgumentException("leftX >= rightX");
		if (values.length < 2)
			throw new IllegalArgumentException("Points count less than 2");
		_pointsCount = values.length;
		_points = new FunctionPoint[_pointsCount];
		double step = Math.abs(rightX - leftX) / (_pointsCount - 1);
		for (int i = 0; i != _pointsCount; ++i) {
			_points[i] = new FunctionPoint(leftX, values[i]);
			leftX += step;
		}
	}

	public ArrayTabulatedFunction(FunctionPoint[] values)
			throws IllegalArgumentException {
		if (values.length < 2)
			throw new IllegalArgumentException("Points count less than 2");
		_points = new FunctionPoint[values.length];
		_pointsCount = values.length;
		_points[0] = new FunctionPoint(values[0]);
		for (int i = 1; i != values.length; ++i) {
			if (values[i - 1].getX() < values[i].getX())
				_points[i] = new FunctionPoint(values[i]);
			else
				throw new IllegalArgumentException(
						"The passed function point values are not ordered by ascending abscissa ["
								+ (i - 1) + "]: " + values[i - 1] +
								" >= [" + i + "]: " + values[i]);
		}
	}

	// Methods
	public double getLeftDomainBorder() {
		return _points[0].getX();
	}

	public double getRightDomainBorder() {
		return _points[_points.length - 1].getX();
	}

	public double getFunctionValue(double x) {
		if (isEmpty() ||
				x < _points[0].getX() ||
				x > _points[_pointsCount - 1].getX())
			return Double.NaN;
		if (_points[_pointsCount - 1].getX() == x)
			return _points[_pointsCount - 1].getY();
		FunctionPoint leftBound = new FunctionPoint(),
				rightBound = new FunctionPoint();
		for (int i = _pointsCount - 1; i != 0; --i) {
			if (x >= _points[i - 1].getX()) {
				leftBound = _points[i - 1];
				rightBound = _points[i];
				break;
			}
		}
		return getLinearFunctionValue(leftBound, rightBound, x);
	}

	// Acces
	public int getPointsCount() {
		return _pointsCount;
	}

	public FunctionPoint getPoint(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		return new FunctionPoint(_points[index]);
	}

	public void setPoint(int index, FunctionPoint point)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		double leftBound = index == 0 ? Double.NEGATIVE_INFINITY : _points[index - 1].getX();
		double rightBound = index == _pointsCount - 1 ? Double.POSITIVE_INFINITY : _points[index + 1].getX();
		if (point.getX() >= leftBound && point.getX() <= rightBound)
			_points[index] = point;
		else
			throw new InappropriateFunctionPointException();
	}

	public double getPointX(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		return _points[index].getX();
	}

	public void setPointX(int index, double x)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		setPoint(index, new FunctionPoint(x, _points[index].getY()));
	}

	public double getPointY(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		return _points[index].getY();
	}

	public void setPointY(int index, double y) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		_points[index].setY(y);
	}

	public void deletePoint(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException();
		if (_pointsCount < 3)
			throw new IllegalStateException("Number os points is less than 3");
		System.arraycopy(_points, index + 1, _points, index, _pointsCount - index - 1);
		_points[_pointsCount - 1] = null;
		--_pointsCount;
	}

	public void addPoint(FunctionPoint point) throws InappropriateFunctionPointException {
		if (isContains(point.getX()))
			throw new InappropriateFunctionPointException();
		if (_pointsCount == _points.length)
			realloc();
		int place = getPlaceForPoint(point.getX());
		System.arraycopy(_points, place, _points, place + 1, _pointsCount - place);
		_points[place] = point;
		++_pointsCount;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer("{ ");
		for (int i = 0; i != _pointsCount; ++i) {
			buffer.append(_points[i].toString());
			if (i != _pointsCount - 1) {
				buffer.append(", ");
			}
		}
		buffer.append(" }");
		return buffer.toString();
	}

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o instanceof ArrayTabulatedFunction) {
			ArrayTabulatedFunction aTabulatedFunction = (ArrayTabulatedFunction) o;
			if (_pointsCount != aTabulatedFunction._pointsCount)
				return false;
			for (int i = 0; i != _pointsCount; ++i) {
				if (!_points[i].equals(aTabulatedFunction._points[i]))
					return false;
			}
			return true;
		}
		if (o instanceof TabulatedFunction) {
			TabulatedFunction tabulatedFunction = (TabulatedFunction) o;
			if (tabulatedFunction.getPointsCount() != _pointsCount)
				return false;
			for (int i = 0; i != _pointsCount; ++i) {
				if (!_points[i].equals(tabulatedFunction.getPoint(i)))
					return false;
			}
			return true;
		}
		return false;
	}

	public int hashCode() {
		int result = 0;
		for (FunctionPoint functionPoint : _points) {
			result ^= functionPoint.hashCode();
		}
		result ^= _pointsCount;
		return result;
	}

	public Object clone() {
		return new ArrayTabulatedFunction(_points);
	}
}
