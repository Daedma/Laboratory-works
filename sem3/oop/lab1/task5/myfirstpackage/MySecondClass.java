package myfirstpackage;

public class MySecondClass {
  private int a;
  private int b;

  public MySecondClass(int A, int B) {
    a = A;
    b = B;
  }

  public int multiply() {
    return a * b;
  }

  public int getA() {
    return a;
  }

  public int getB() {
    return b;
  }

  public void setA(int A) {
    a = A;
  }

  public void setB(int B) {
    b = B;
  }
}
