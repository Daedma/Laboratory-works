package functions;

import java.io.Serializable;

public class FunctionPoint implements Serializable {
	// Members
	private double _x;
	private double _y;

	// Constructors
	public FunctionPoint(double x, double y) {
		_x = x;
		_y = y;
	}

	public FunctionPoint(FunctionPoint point) {
		_x = point._x;
		_y = point._y;
	}

	public FunctionPoint() {
		this(0, 0);
	}

	// Acces metods
	public double getX() {
		return _x;
	}

	public double getY() {
		return _y;
	}

	public void setX(double value) {
		_x = value;
	}

	public void setY(double value) {
		_y = value;
	}

	public void setXY(double x, double y) {
		_x = x;
		_y = y;
	}

	public String toString() {
		return "(" + Double.toString(_x) + "; "
				+ Double.toString(_y) + ")";
	}

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof FunctionPoint))
			return false;
		FunctionPoint tmp = (FunctionPoint) o;
		return tmp._x == _x && tmp._y == _y;
	}

	public int hashCode() {
		long xBits = Double.doubleToLongBits(_x);
		long yBits = Double.doubleToLongBits(_y);
		int xBitsLeftPart = (int) (xBits >> 32);
		int xBitsRightPart = (int) xBits;
		int yBitsLeftPart = (int) (yBits >> 32);
		int yBitsRightPart = (int) yBits;
		return xBitsLeftPart ^ xBitsRightPart
				^ yBitsLeftPart ^ yBitsRightPart;
	}

	public Object clone() {
		return new FunctionPoint(_x, _y);
	}
}
