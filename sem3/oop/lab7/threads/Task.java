package threads;

import functions.Function;
import functions.Functions;

public class Task {
	Function function;

	double left, right;

	double step = Double.MIN_NORMAL;

	int taskCount;

	public boolean isInitialize() {
		return function != null;
	}

	public double integrate() {
		return Functions.integrate(function, left, right, step);
	}

	public void setFunction(Function function) {
		this.function = function;
	}

	public Function getFunction() {
		return function;
	}

	public void setLeft(double left) {
		this.left = left;
	}

	public double getLeft() {
		return left;
	}

	public void setRight(double right) {
		this.right = right;
	}

	public double getRight() {
		return right;
	}

	public void setStep(double step) {
		this.step = step;
	}

	public double getStep() {
		return step;
	}

	public void setTaskCount(int taskCount) {
		this.taskCount = taskCount;
	}

	public int getTaskCount() {
		return taskCount;
	}
}
