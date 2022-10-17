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
}
