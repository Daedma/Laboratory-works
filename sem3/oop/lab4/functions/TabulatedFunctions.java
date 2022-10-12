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
import java.io.Reader;

final public class TabulatedFunctions {
	private TabulatedFunctions() {
	}

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
		return new ArrayTabulatedFunction(values);
	}

	public static void outputTabulatedFunction(TabulatedFunction function, OutputStream out)
			throws IOException {
		DataOutputStream dataOut = new DataOutputStream(out);
		dataOut.writeInt(function.getPointsCount());
		for (int i = 0; i != function.getPointsCount(); ++i) {
			dataOut.writeDouble(function.getPointX(i));
			dataOut.writeDouble(function.getPointY(i));
		}
	}

	public static TabulatedFunction inputTabulatedFunction(InputStream in)
			throws IOException {
		DataInputStream dataIn = new DataInputStream(in);
		FunctionPoint[] values = new FunctionPoint[dataIn.readInt()];
		for (int i = 0; i != values.length; ++i) {
			values[i] = new FunctionPoint(dataIn.readDouble(), dataIn.readDouble());
		}
		return new ArrayTabulatedFunction(values);
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
	}

	public static TabulatedFunction readTabulatedFunction(Reader in)
			throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(in);
		tokenizer.parseNumbers();
		FunctionPoint[] values = new FunctionPoint[(int) tokenizer.nval];
		for (int i = 0; i != values.length; ++i) {
			tokenizer.nextToken();
			values[i] = new FunctionPoint(tokenizer.nval, Double.NaN);
			tokenizer.nextToken();
			values[i].setY(tokenizer.nval);
		}
		return new ArrayTabulatedFunction(values);
	}
}
