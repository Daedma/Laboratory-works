public class FunctionPointT {
	private double x, y;

	FunctionPointT() {
		x = 0;
		y = 0;
	}

	FunctionPointT(double x, double y) {
		this.x = x;
		this.y = y;
	}

	public Double getX() {
		return Double.valueOf(x);
	}

	public Double getY() {
		return Double.valueOf(y);
	}

	public void setX(Double x) {
		this.x = x;
	}

	public void setY(Double y) {
		this.y = y;
	}
}
