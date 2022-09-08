class MySecondClass {
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

class MyFirstClass {
  public static void main(String[] s) {
    // <Создание и инициализация объекта “o” типа MySecondClass>;
    MySecondClass o = new MySecondClass(14, 88);
    int i, j;
    for (i = 1; i <= 8; i++) {
      for (j = 1; j <= 8; j++) {
        // o.<Метод установки значения первого числового поля>(i);
        o.setA(i);
        // o.<Метод установки значения второго числового поля>(j);
        o.setB(j);
        System.out.print(o.multiply());
        System.out.print(" ");
      }
      System.out.println();
    }

  }
}
