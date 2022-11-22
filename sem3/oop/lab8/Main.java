import functions.Function;
import functions.FunctionPoint;
import functions.TabulatedFunction;
import functions.TabulatedFunctions;
import functions.basic.Sin;
import functions.*;
import functions.basic.*;

public class Main {
	public static void main(String[] args) {
		factoryTest();
		reflectionTest();

	}

	private static void factoryTest() {
		Function f = new Cos();
		TabulatedFunction tf;
		tf = TabulatedFunctions.tabulate(f, 0, Math.PI, 11);
		System.out.println(tf.getClass());
		TabulatedFunctions
				.setTabulatedFunctionFactory(new LinkedListTabulatedFunction.LinkedListTabulatedFunctionFactory());
		tf = TabulatedFunctions.tabulate(f, 0, Math.PI, 11);
		System.out.println(tf.getClass());
		TabulatedFunctions.setTabulatedFunctionFactory(new ArrayTabulatedFunction.ArrayTabulatedFunctionFactory());
		tf = TabulatedFunctions.tabulate(f, 0, Math.PI, 11);
		System.out.println(tf.getClass());
	}

	private static void reflectionTest() {
		TabulatedFunction f;

		f = TabulatedFunctions.createTabulatedFunction(
				ArrayTabulatedFunction.class, 0, 10, 3);
		System.out.println(f.getClass());
		System.out.println(f);

		f = TabulatedFunctions.createTabulatedFunction(
				ArrayTabulatedFunction.class, 0, 10, new double[] { 0, 10 });
		System.out.println(f.getClass());
		System.out.println(f);

		f = TabulatedFunctions.createTabulatedFunction(
				LinkedListTabulatedFunction.class,
				new FunctionPoint[] {
						new FunctionPoint(0, 0),
						new FunctionPoint(10, 10)
				});
		System.out.println(f.getClass());
		System.out.println(f);

		f = TabulatedFunctions.tabulate(
				LinkedListTabulatedFunction.class, new Sin(), 0, Math.PI, 11);
		System.out.println(f.getClass());
		System.out.println(f);
	}
}
