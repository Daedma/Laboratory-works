package gui;

//TODO: CallRedraw & modified
import functions.ArrayTabulatedFunction;
import functions.FunctionPoint;
import functions.FunctionPointIndexOutOfBoundsException;
import functions.InappropriateFunctionPointException;
import functions.TabulatedFunction;
import functions.TabulatedFunctions;
import functions.Function;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import org.json.simple.JSONArray;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;

public class TabulatedFunctionDoc implements TabulatedFunction {
	// Members
	private TabulatedFunction _function;
	private String _fileName;
	private boolean _changed;
	private Controller _contoller;

	// Methods
	public void CallRedraw() {
		_contoller.redraw();
	}

	public void registerRedrawFunctionController(Controller controller) {
		_contoller = controller;
		CallRedraw();
	}

	public boolean isModified() {
		return _changed;
	}

	public boolean isFileNameAssigned() {
		return _fileName != null && !_fileName.isEmpty();
	}

	public void newFunction(double leftX, double rightX, int pointsCount) throws IllegalArgumentException {
		_function = new ArrayTabulatedFunction(leftX, rightX, pointsCount);
		_changed = true;
		if (_contoller != null)
			CallRedraw();
	}

	public void tabulateFunction(Function function, double leftX, double rightX, int pointsCount)
			throws IllegalArgumentException {
		_function = TabulatedFunctions.tabulate(function, leftX, rightX, pointsCount);
		_changed = true;
		CallRedraw();
	}

	public void saveFunctionAs(String fileName) throws IOException {
		System.out.println(fileName);
		_fileName = fileName;
		JSONObject tabulatedFunctionData = new JSONObject();
		JSONArray points = new JSONArray();
		for (int i = 0; i != _function.getPointsCount(); ++i) {
			JSONObject point = new JSONObject();
			point.put("x", _function.getPointX(i));
			point.put("y", _function.getPointY(i));
			points.add(point);
		}
		tabulatedFunctionData.put("points", points);
		FileWriter file = new FileWriter(fileName);
		tabulatedFunctionData.writeJSONString(file);
		file.close();
		_changed = false;
	}

	public void loadFunction(String fileName) throws IOException, Exception {
		_fileName = fileName;
		FileReader file = new FileReader(fileName);
		JSONParser parser = new JSONParser();
		JSONObject jsonObject = (JSONObject) parser.parse(file);
		JSONArray jsonPoints = (JSONArray) jsonObject.get("points");
		FunctionPoint[] points = new FunctionPoint[jsonPoints.size()];
		for (int i = 0; i != points.length; ++i) {
			JSONObject point = (JSONObject) jsonPoints.get(i);
			points[i] = new FunctionPoint((Double) point.get("x"), (Double) point.get("y"));
		}
		_function = new ArrayTabulatedFunction(points);
		file.close();
		_changed = false;
		CallRedraw();
	}

	public void saveFunction() throws IOException {
		saveFunctionAs(_fileName);
	}

	public int getPointsCount() {
		return _function.getPointsCount();
	}

	public FunctionPoint getPoint(int index) throws FunctionPointIndexOutOfBoundsException {
		return _function.getPoint(index);
	}

	public double getFunctionValue(double x) {
		return _function.getFunctionValue(x);
	}

	public double getLeftDomainBorder() {
		return _function.getLeftDomainBorder();
	}

	public double getRightDomainBorder() {
		return _function.getRightDomainBorder();
	}

	public void setPoint(int index, FunctionPoint point)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException {
		_changed = true;
		_function.setPoint(index, point);
		CallRedraw();
	}

	public double getPointX(int index) throws FunctionPointIndexOutOfBoundsException {
		return _function.getPointX(index);
	}

	public void setPointX(int index, double x) throws FunctionPointIndexOutOfBoundsException,
			InappropriateFunctionPointException {
		_function.setPointX(index, x);
		_changed = true;
		CallRedraw();
	}

	public double getPointY(int index) throws FunctionPointIndexOutOfBoundsException {
		return _function.getPointY(index);
	}

	public void setPointY(int index, double y) throws FunctionPointIndexOutOfBoundsException {
		_function.setPointY(index, y);
		_changed = true;
		CallRedraw();
	}

	public void deletePoint(int index) throws FunctionPointIndexOutOfBoundsException {
		_function.deletePoint(index);
		_changed = true;
		CallRedraw();
	}

	public void addPoint(FunctionPoint point) throws InappropriateFunctionPointException {
		_function.addPoint(point);
		_changed = true;
		CallRedraw();
	}

	public Object clone() {
		TabulatedFunctionDoc tDoc = new TabulatedFunctionDoc();
		tDoc._function = (TabulatedFunction) _function.clone();
		tDoc._changed = _changed;
		tDoc._fileName = _fileName;
		return tDoc;
	}

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof TabulatedFunction))
			return false;
		TabulatedFunction rhs = (TabulatedFunction) o;
		return _function.equals(rhs);
	}

	public String toString() {
		return _function.toString();
	}

	public int hashCode() {
		return _function.hashCode();
	}
}
