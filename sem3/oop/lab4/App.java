import functions.*;

public class App {
	public static void main(String[] args) {
		TabulatedFunction cube = new LinkedListTabulatedFunction(-6., 6.,
				new double[] { -216, -125, -64, -27,
						-8, -1, 0, 1, 8, 27, 64, 125, 216 });
		System.out.println("Initial points:");
		printPoints(cube);
		try {
			cube.addPoint(new FunctionPoint(1.5, 3.375));
			cube.deletePoint(6);
			cube.setPointY(8, 54);
			cube.setPoint(3, new FunctionPoint(-3.5, -42.875));
			cube.setPointX(0, 12); // throw exception
		} catch (FunctionPointIndexOutOfBoundsException e) {
			System.err.println("An exception was thrown: " + e.getMessage());
		} catch (InappropriateFunctionPointException e) {
			System.err.println("An exception was thrown: " + e.getMessage());
		}
		System.out.println("...After modification:");
		printPoints(cube);
	}

	private static void printPoints(TabulatedFunction function) {
		for (double i = function.getLeftDomainBorder(); i <= 7.; i += 0.5) {
			System.out.printf("x: %f\ty: %f\n", i, function.getFunctionValue(i));
		}
	}
}
