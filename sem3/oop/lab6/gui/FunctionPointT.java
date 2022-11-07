package gui;

public class FunctionPointT {
	private double x, y;

	public FunctionPointT() {
		x = 0;
		y = 0;
	}

	public FunctionPointT(double x, double y) {
		this.x = x;
		this.y = y;
	}

	public Double getX() {
		return Double.valueOf(x);
	}

	public Double getY() {
		return Double.valueOf(y);
	}

	public void setX(Double value) {
		this.x = x;
	}

	public void setY(Double value) {
		this.y = y;
	}
}
