package functions;

public class TabulatedFunction {
    // Members
    private FunctionPoint[] _points;
    private int _pointsCount;

    private static final double REALLOC_MULTIPLIER = 2.0;

    // Private methods
    private double getLinearFunctionValue(FunctionPoint firstPoint, FunctionPoint secondPoint, double x) {
        return (x - firstPoint.getX()) * (secondPoint.getY() - firstPoint.getY()) /
                (secondPoint.getX() - firstPoint.getX())
                + firstPoint.getY();
    }

    private boolean isEmpty() {
        return _pointsCount == 0;
    }

    private int getPlaceForPoint(double x) {
        for (int i = _pointsCount - 1; i >= 0; --i) {
            if (x > _points[i].getX())
                return i + 1;
        }
        return 0;
    }

    private void realloc() {
        FunctionPoint[] buff = new FunctionPoint[(int) (_points.length * REALLOC_MULTIPLIER)];
        System.arraycopy(_points, 0, buff, 0, _pointsCount);
        _points = buff;
    }

    // Constructors
    public TabulatedFunction(double leftX, double rightX, int pointsCount) {
        _pointsCount = pointsCount;
        _points = new FunctionPoint[pointsCount];
        double step = Math.abs(rightX - leftX + 1.) / pointsCount;
        for (int i = 0; i != pointsCount; ++i) {
            _points[i] = new FunctionPoint(leftX, 0);
            leftX += step;
        }
    }

    public TabulatedFunction(double leftX, double rightX, double[] values) {
        _pointsCount = values.length;
        _points = new FunctionPoint[_pointsCount];
        double step = Math.abs(rightX - leftX + 1.) / _pointsCount;
        for (int i = 0; i != _pointsCount; ++i) {
            _points[i] = new FunctionPoint(leftX, values[i]);
            leftX += step;
        }
    }

    // Methods
    public double getLeftDomainBorder() {
        return _points[0].getX();
    }

    public double getFunctionValue(double x) {
        if (isEmpty() ||
                x < _points[0].getX() ||
                x > _points[_pointsCount - 1].getX())
            return Double.NaN;
        if (_pointsCount == 1)
            return _points[0].getY();
        if (_points[_pointsCount - 1].getX() == x)
            return _points[_pointsCount - 1].getY();
        FunctionPoint leftBound = new FunctionPoint(),
                rightBound = new FunctionPoint();
        for (int i = _pointsCount - 1; i != 0; --i) {
            if (x >= _points[i - 1].getX()) {
                leftBound = _points[i - 1];
                rightBound = _points[i];
                break;
            }
        }
        return getLinearFunctionValue(leftBound, rightBound, x);
    }

    // Acces
    public int getPointsCount() {
        return _pointsCount;
    }

    public FunctionPoint getPoint(int index) {
        return _points[index];
    }

    public void setPoint(int index, FunctionPoint point) {
        if (isEmpty())
            return;
        double leftBound = index == 0 ? Double.NEGATIVE_INFINITY : _points[index - 1].getX();
        double rightBound = index == _pointsCount - 1 ? Double.POSITIVE_INFINITY : _points[index + 1].getX();
        if (point.getX() >= leftBound && point.getX() <= rightBound)
            _points[index] = point;
    }

    public double getPointX(int index) {
        return _points[index].getX();
    }

    public void setPointX(int index, double x) {
        setPoint(index, new FunctionPoint(x, _points[index].getY()));
    }

    public double getPointY(int index) {
        return _points[index].getY();
    }

    public void setPointY(int index, double y) {
        _points[index].setY(y);
    }

    public void deletePoint(int index) {
        System.arraycopy(_points, index + 1, _points, index, _pointsCount - index - 1);
        _points[_pointsCount - 1] = null;
        --_pointsCount;
    }

    public void addPoint(FunctionPoint point) {
        if (_pointsCount == _points.length)
            realloc();
        int place = getPlaceForPoint(point.getX());
        System.arraycopy(_points, place, _points, place + 1, _pointsCount - place);
        _points[place] = point;
        ++_pointsCount;
    }
}
