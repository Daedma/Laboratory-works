package functions;

import java.io.Serializable;

public class LinkedListTabulatedFunction implements TabulatedFunction, Serializable {
	// Classes
	private class FunctionNode implements Serializable {
		public FunctionPoint _data;
		public FunctionNode _prev;
		public FunctionNode _next;

		public FunctionNode() {
			_prev = this;
			_next = this;
		}

		public FunctionNode(FunctionPoint data, FunctionNode prev, FunctionNode next) {
			_data = data;
			_prev = prev;
			_next = next;
		}

		public FunctionNode(FunctionPoint data) {
			this(data, null, null);
		}

		public FunctionNode(FunctionNode rhs) {
			_data = rhs._data;
			_prev = rhs._prev;
			_next = rhs._next;
		}

		// emplace node before this
		public void emplaceBefore(FunctionNode node) {
			node._next = this;
			node._prev = _prev;
			node._prev._next = node;
			_prev = node;
		}

		public void remove() {
			_prev._next = _next;
			_next._prev = _prev;
			_prev = _next = null;
		}
	}

	// Members
	private FunctionNode _head = new FunctionNode();
	private FunctionNode _lastReadWriteNode = _head;
	private int _lastReadWriteNodeIndex = 0;
	private int _size = 0;

	// Private methods
	private int minAbs(int rhs, int lhs) {
		return Math.abs(rhs) < Math.abs(lhs) ? rhs : lhs;
	}

	private FunctionNode translate(FunctionNode start, int offset) {
		FunctionNode result = start;
		if (offset < 0) {
			if (_head == start)
				result = result._prev;
			for (int i = 0; i != offset; --i) {
				if (result == _head)
					++i;
				result = result._prev;
			}
		} else {
			if (_head == start)
				result = result._next;
			for (int i = 0; i != offset; ++i) {
				if (result == _head)
					--i;
				result = result._next;
			}
		}
		return result;
	}

	private FunctionNode getNodeByIndex(int index) {
		int minDistanceLast = minAbs(index - _lastReadWriteNodeIndex,
				index - _size - _lastReadWriteNodeIndex);
		int minDistanceHead = minAbs(index, index - _size);
		int minDistance = minAbs(minDistanceHead, minDistanceLast);
		FunctionNode start = minDistance == minDistanceLast ? _lastReadWriteNode : _head._next;
		_lastReadWriteNode = translate(start, minDistance);
		_lastReadWriteNodeIndex = index;
		return _lastReadWriteNode;
	}

	private FunctionNode addNodeToTail() {
		FunctionNode newNode = new FunctionNode();
		_head.emplaceBefore(newNode);
		++_size;
		return newNode;
	}

	private FunctionNode addNodeByIndex(int index) {
		FunctionNode place = getNodeByIndex(index);
		FunctionNode newNode = new FunctionNode();
		place.emplaceBefore(newNode);
		++_size;
		_lastReadWriteNodeIndex += _lastReadWriteNodeIndex >= index ? 1 : 0;
		return newNode;
	}

	private FunctionNode deleteNodeByIndex(int index) {
		FunctionNode nodeToBeDeleted = getNodeByIndex(index);
		if (_lastReadWriteNode == nodeToBeDeleted) {
			_lastReadWriteNode = _lastReadWriteNode._next;
			_lastReadWriteNodeIndex += _lastReadWriteNode == _head ? -_lastReadWriteNodeIndex : 0;
		}
		nodeToBeDeleted.remove();
		--_size;
		return nodeToBeDeleted;
	}

	private boolean isEmpty() {
		return _size == 0;
	}

	private boolean isValidIndex(int index) {
		return index >= 0 && index < _size;
	}

	private boolean isContains(double x) {
		for (FunctionNode i = _head._next; i != _head; i = i._next) {
			if (i._data.getX() == x)
				return true;
		}
		return false;
	}

	private double getLinearFunctionValue(FunctionPoint firstPoint, FunctionPoint secondPoint, double x) {
		return (x - firstPoint.getX()) * (secondPoint.getY() - firstPoint.getY()) /
				(secondPoint.getX() - firstPoint.getX())
				+ firstPoint.getY();
	}

	private FunctionPoint getFirstPoint() {
		return _head._next._data;
	}

	private FunctionPoint getLastPoint() {
		return _head._prev._data;
	}

	// Constructors
	public LinkedListTabulatedFunction(double leftX, double rightX, int pointsCount) throws IllegalArgumentException {
		if (leftX >= rightX)
			throw new IllegalArgumentException("leftX >= rightX");
		if (pointsCount < 2)
			throw new IllegalArgumentException("points count less than 2");
		double step = Math.abs(rightX - leftX + 1.) / pointsCount;
		for (int i = 0; i != pointsCount; ++i) {
			addNodeToTail()._data = new FunctionPoint(leftX, 0);
			leftX += step;
		}
	}

	public LinkedListTabulatedFunction(double leftX, double rightX, double[] values) throws IllegalArgumentException {
		if (leftX >= rightX)
			throw new IllegalArgumentException("leftX >= rightX");
		if (values.length < 2)
			throw new IllegalArgumentException("points count less than 2");
		double step = Math.abs(rightX - leftX + 1.) / values.length;
		for (int i = 0; i != values.length; ++i) {
			addNodeToTail()._data = new FunctionPoint(leftX, values[i]);
			leftX += step;
		}
	}

	public LinkedListTabulatedFunction(FunctionPoint[] values)
			throws IllegalArgumentException {
		if (values.length < 2)
			throw new IllegalArgumentException("Points count less than 2");
		addNodeToTail()._data = new FunctionPoint(values[0]);
		for (int i = 1; i != values.length; ++i) {
			if (values[i - 1].getX() < values[i].getX())
				addNodeToTail()._data = new FunctionPoint(values[i]);
			else
				throw new IllegalArgumentException(
						"The passed function point values are not ordered by ascending abscissa ["
								+ (i - 1) + "]: " + values[i - 1] +
								" >= [" + i + "]: " + values[i]);
		}
	}

	// Methods
	public double getLeftDomainBorder() {
		return getFirstPoint().getX();
	}

	public double getRightDomainBorder() {
		return getLastPoint().getX();
	}

	public double getFunctionValue(double x) {
		if (isEmpty() ||
				x < getFirstPoint().getX() ||
				x > getLastPoint().getX())
			return Double.NaN;
		if (getLastPoint().getX() == x)
			return getLastPoint().getY();
		FunctionPoint leftBound = new FunctionPoint(),
				rightBound = new FunctionPoint();
		for (FunctionNode i = _head._prev; i != _head; i = i._prev) {
			if (x >= i._prev._data.getX()) {
				leftBound = i._prev._data;
				rightBound = i._data;
				break;
			}
		}
		return getLinearFunctionValue(leftBound, rightBound, x);
	}

	// Acces
	public int getPointsCount() {
		return _size;
	}

	public FunctionPoint getPoint(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		return new FunctionPoint(getNodeByIndex(index)._data);
	}

	public void setPoint(int index, FunctionPoint point)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		FunctionNode pointWrite = getNodeByIndex(index);
		double leftBound = index == 0 ? Double.NEGATIVE_INFINITY : pointWrite._prev._data.getX();
		double rightBound = index == _size - 1 ? Double.POSITIVE_INFINITY : pointWrite._next._data.getX();
		if (point.getX() >= leftBound && point.getX() <= rightBound)
			pointWrite._data = point;
		else
			throw new InappropriateFunctionPointException(
					"X must be enclosed between " + leftBound + " and " + rightBound + " (x = " + point.getX() + ") ");
	}

	public double getPointX(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		return getNodeByIndex(index)._data.getX();
	}

	public void setPointX(int index, double x)
			throws FunctionPointIndexOutOfBoundsException, InappropriateFunctionPointException {
		setPoint(index, new FunctionPoint(x, getNodeByIndex(index)._data.getY()));
	}

	public double getPointY(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		return getNodeByIndex(index)._data.getY();
	}

	public void setPointY(int index, double y) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		getNodeByIndex(index)._data.setY(y);
	}

	public void deletePoint(int index) throws FunctionPointIndexOutOfBoundsException {
		if (!isValidIndex(index))
			throw new FunctionPointIndexOutOfBoundsException(
					"invalid index " + index + " (index must be from 0 to " + (_size - 1) + " )");
		if (_size < 3)
			throw new IllegalStateException("Number os points is less than 3");
		deleteNodeByIndex(index);
	}

	public void addPoint(FunctionPoint point) throws InappropriateFunctionPointException {
		if (isContains(point.getX()))
			throw new InappropriateFunctionPointException(
					"Point with this abcis already exists (x = " + point.getX() + ")");
		for (FunctionNode i = _head; i != _head._next; i = i._prev) {
			if (i._prev._data.getX() < point.getX()) {
				i.emplaceBefore(new FunctionNode(point));
				++_size;
				return;
			}
		}
		_head._next.emplaceBefore(new FunctionNode(point));
		++_size;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer("{ ");
		for (FunctionNode i = _head._next; i != _head; i = i._next) {
			buffer.append(i._data.toString());
			if (i._next != _head) {
				buffer.append(", ");
			}
		}
		buffer.append(" }");
		return buffer.toString();
	}

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (o instanceof LinkedListTabulatedFunction) {
			LinkedListTabulatedFunction lTabulatedFunction = (LinkedListTabulatedFunction) o;
			if (_size != lTabulatedFunction._size)
				return false;
			for (FunctionNode i = _head._next,
					j = lTabulatedFunction._head._next; i != _head; i = i._next, j = j._next) {
				if (!i._data.equals(j._data))
					return false;
			}
			return true;
		}
		if (o instanceof TabulatedFunction) {
			TabulatedFunction tabulatedFunction = (TabulatedFunction) o;
			if (_size != tabulatedFunction.getPointsCount())
				return false;
			FunctionNode i = _head._next;
			for (int j = 0; j != _size; ++j, i = i._next) {
				if (!tabulatedFunction.getPoint(j).equals(i._data))
					return false;
			}
			return true;
		}
		return false;
	}

	public int hashCode() {
		int result = 0;
		for (FunctionNode i = _head._next; i != _head; i = i._next) {
			result ^= i._data.hashCode();
		}
		result ^= _size;
		return result;
	}

	public Object clone() {
		FunctionPoint[] points = new FunctionPoint[_size];
		int j = 0;
		for (FunctionNode i = _head._next; i != _head; i = i._next, ++j) {
			points[j] = new FunctionPoint(i._data);
		}
		return new LinkedListTabulatedFunction(points);
	}
}
