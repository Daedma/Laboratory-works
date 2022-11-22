package functions;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StreamTokenizer;
import java.io.InputStream;
import java.io.Writer;
import java.lang.reflect.Array;
import java.io.Reader;

final public class TabulatedFunctions {

	private static TabulatedFunctionFactory tabulatedFunctionFactory = new ArrayTabulatedFunction.ArrayTabulatedFunctionFactory();

	private TabulatedFunctions() {
	}

	// Creating with factory
	public static void setTabulatedFunctionFactory(TabulatedFunctionFactory tabulatedFunctionFactory) {
		TabulatedFunctions.tabulatedFunctionFactory = tabulatedFunctionFactory;
	}

	public static TabulatedFunction createTabulatedFunction(double leftX, double rightX, int pointsCount)
			throws IllegalArgumentException {
		return tabulatedFunctionFactory.createTabulatedFunction(leftX, rightX, pointsCount);
	}

	public static TabulatedFunction createTabulatedFunction(double leftX, double rightX, double[] values)
			throws IllegalArgumentException {
		return tabulatedFunctionFactory.createTabulatedFunction(leftX, rightX, values);
	}

	public static TabulatedFunction createTabulatedFunction(FunctionPoint[] values) throws IllegalArgumentException {
		return tabulatedFunctionFactory.createTabulatedFunction(values);
	}

	// Creating with reflection
	public static TabulatedFunction createTabulatedFunction(Class<? extends TabulatedFunction> fClass, double leftX,
			double rightX, int pointsCount)
			throws IllegalArgumentException {
		try {
			return fClass.getDeclaredConstructor(Double.TYPE, Double.TYPE, Integer.TYPE).newInstance(leftX, rightX,
					pointsCount);
		} catch (Exception e) {
			throw new IllegalArgumentException(e.getLocalizedMessage(), e);
		}
	}

	public static TabulatedFunction createTabulatedFunction(Class<? extends TabulatedFunction> fClass, double leftX,
			double rightX, double[] values)
			throws IllegalArgumentException {
		try {
			return fClass.getDeclaredConstructor(Double.TYPE, Double.TYPE, Double.TYPE.arrayType()).newInstance(
					leftX,
					rightX, values);
		} catch (Exception e) {
			throw new IllegalArgumentException(e.getLocalizedMessage(), e);
		}
	}

	public static TabulatedFunction createTabulatedFunction(Class<? extends TabulatedFunction> fClass,
			FunctionPoint[] values) throws IllegalArgumentException {
		try {
			return fClass.getDeclaredConstructor(FunctionPoint.class.arrayType()).newInstance((Object) values);
		} catch (Exception e) {
			throw new IllegalArgumentException(e.getLocalizedMessage(), e);
		}
	}

	// Other
	public static TabulatedFunction tabulate(Function function, double leftX, double rightX, int pointsCount)
			throws IllegalArgumentException {
		if (function.getLeftDomainBorder() > leftX || function.getRightDomainBorder() < rightX)
			throw new IllegalArgumentException("Specified tab limits are outside the scope of the function");
		FunctionPoint[] values = new FunctionPoint[pointsCount];
		double step = Math.abs(rightX - leftX + 1.) / pointsCount;
		for (int i = 0; i != pointsCount; ++i) {
			values[i] = new FunctionPoint(leftX, function.getFunctionValue(leftX));
			leftX += step;
		}
		return createTabulatedFunction(values);
	}

	public static TabulatedFunction tabulate(Class<? extends TabulatedFunction> fClass, Function function, double leftX,
			double rightX, int pointsCount)
			throws IllegalArgumentException {
		if (function.getLeftDomainBorder() > leftX || function.getRightDomainBorder() < rightX)
			throw new IllegalArgumentException("Specified tab limits are outside the scope of the function");
		FunctionPoint[] values = new FunctionPoint[pointsCount];
		double step = Math.abs(rightX - leftX + 1.) / pointsCount;
		for (int i = 0; i != pointsCount; ++i) {
			values[i] = new FunctionPoint(leftX, function.getFunctionValue(leftX));
			leftX += step;
		}
		return createTabulatedFunction(fClass, values);
	}

	public static void outputTabulatedFunction(TabulatedFunction function, OutputStream out)
			throws IOException {
		DataOutputStream dataOut = new DataOutputStream(out);
		dataOut.writeInt(function.getPointsCount());
		for (int i = 0; i != function.getPointsCount(); ++i) {
			dataOut.writeDouble(function.getPointX(i));
			dataOut.writeDouble(function.getPointY(i));
		}
		dataOut.flush();
	}

	public static TabulatedFunction inputTabulatedFunction(InputStream in)
			throws IOException {
		DataInputStream dataIn = new DataInputStream(in);
		FunctionPoint[] values = new FunctionPoint[dataIn.readInt()];
		for (int i = 0; i != values.length; ++i) {
			values[i] = new FunctionPoint(dataIn.readDouble(), dataIn.readDouble());
		}
		return createTabulatedFunction(values);
	}

	public static TabulatedFunction inputTabulatedFunction(Class<? extends TabulatedFunction> fClass, InputStream in)
			throws IOException {
		DataInputStream dataIn = new DataInputStream(in);
		FunctionPoint[] values = new FunctionPoint[dataIn.readInt()];
		for (int i = 0; i != values.length; ++i) {
			values[i] = new FunctionPoint(dataIn.readDouble(), dataIn.readDouble());
		}
		return createTabulatedFunction(fClass, values);
	}

	public static void writeTabulatedFunction(TabulatedFunction function, Writer out)
			throws IOException {
		PrintWriter output = new PrintWriter(new BufferedWriter(out));
		output.print(function.getPointsCount());
		for (int i = 0; i != function.getPointsCount(); ++i) {
			output.print(' ');
			output.print(function.getPointX(i));
			output.print(' ');
			output.print(function.getPointY(i));
		}
		output.flush();
	}

	public static TabulatedFunction readTabulatedFunction(Reader in)
			throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(in);
		tokenizer.parseNumbers();
		tokenizer.nextToken();
		FunctionPoint[] values = new FunctionPoint[(int) tokenizer.nval];
		for (int i = 0; i != values.length; ++i) {
			tokenizer.nextToken();
			values[i] = new FunctionPoint(tokenizer.nval, Double.NaN);
			tokenizer.nextToken();
			values[i].setY(tokenizer.nval);
		}
		return createTabulatedFunction(values);
	}

	public static TabulatedFunction readTabulatedFunction(Class<? extends TabulatedFunction> fClass, Reader in)
			throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(in);
		tokenizer.parseNumbers();
		tokenizer.nextToken();
		FunctionPoint[] values = new FunctionPoint[(int) tokenizer.nval];
		for (int i = 0; i != values.length; ++i) {
			tokenizer.nextToken();
			values[i] = new FunctionPoint(tokenizer.nval, Double.NaN);
			tokenizer.nextToken();
			values[i].setY(tokenizer.nval);
		}
		return createTabulatedFunction(fClass, values);
	}
}
