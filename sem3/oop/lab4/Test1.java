import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;

import functions.*;
import functions.basic.Cos;
import functions.basic.Exp;
import functions.basic.Log;
import functions.basic.Sin;

public class Test1 {
	public static void main(String[] args) {
		try {
			Sin sin = new Sin();
			Cos cos = new Cos();
			// Default sin and cos
			System.out.print("Sin: ");
			printFunction(sin, 0, 2 * Math.PI, 0.1);
			System.out.print("Cos: ");
			printFunction(cos, 0, 2 * Math.PI, 0.1);
			// Tabulated
			TabulatedFunction tabulatedSin = TabulatedFunctions.tabulate(sin, 0, 2 * Math.PI, 10);
			TabulatedFunction tabulatedCos = TabulatedFunctions.tabulate(cos, 0, 2 * Math.PI, 10);
			System.out.print("Tabulated sin: ");
			printFunction(tabulatedSin, 0, 2 * Math.PI, 0.1);
			System.out.print("Tabulated cos: ");
			printFunction(tabulatedCos, 0, 2 * Math.PI, 0.1);
			// Sum of square
			Function sumOfSquareCosSin = Functions.sum(Functions.power(tabulatedCos, 2),
					Functions.power(tabulatedSin, 2));
			System.out.print("Cos^2 + Sin^2:");
			printFunction(sumOfSquareCosSin, 0, 2 * Math.PI, 0.1);
			// Exponent
			TabulatedFunction tabulatedExp = TabulatedFunctions.tabulate(new Exp(), 0, 10, 11);
			TabulatedFunctions.writeTabulatedFunction(tabulatedExp, new FileWriter("exp.txt"));
			System.out.print("Initial tabulated exp: ");
			printFunction(tabulatedExp, 0, 10, 1);
			System.out.print("Readed tabulated exp: ");
			printFunction(TabulatedFunctions.readTabulatedFunction(new FileReader("exp.txt")), 0, 10, 1);
			// Logarithm
			TabulatedFunction tabulatedLog = TabulatedFunctions.tabulate(new Log(2), 0, 10, 11);
			TabulatedFunctions.outputTabulatedFunction(tabulatedLog, new FileOutputStream("log.bin"));
			System.out.print("Initial tabulated log2: ");
			printFunction(tabulatedLog, 0, 10, 1);
			System.out.print("Readed tabulated log: ");
			printFunction(TabulatedFunctions.inputTabulatedFunction(new FileInputStream("log.bin")), 0, 10, 11);
		} catch (Exception e) {
			System.err.println(e.getLocalizedMessage());
		}
	}

	private static void printFunction(Function f, double start, double last, double step) {
		for (; start <= last; start += step)
			System.out.printf("(%f, %f) ", start, f.getFunctionValue(start));
		System.out.println(".");
	}
}
