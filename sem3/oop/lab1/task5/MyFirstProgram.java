import myfirstpackage.*;

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
