import functions.ArrayTabulatedFunction;
import functions.FunctionPoint;
import functions.InappropriateFunctionPointException;
import functions.LinkedListTabulatedFunction;

public class Main {
	public static void main(String[] args) {
		try {
			toStringTest();
			equalsTest();
			hashCodeTest();
			cloneTest();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static LinkedListTabulatedFunction cubeLinkedList = new LinkedListTabulatedFunction(-6., 6.,
			new double[] { -216, -125, -64, -27,
					-8, -1, 0, 1, 8, 27, 64, 125, 216 });

	private static LinkedListTabulatedFunction cubeLinkedListCopy = new LinkedListTabulatedFunction(-6., 6.,
			new double[] { -216, -125, -64, -27,
					-8, -1, 0, 1, 8, 27, 64, 125, 216 });

	private static LinkedListTabulatedFunction squareLinkedList = new LinkedListTabulatedFunction(-6., 6.,
			new double[] { 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36 });

	private static ArrayTabulatedFunction cubeArray = new ArrayTabulatedFunction(-6., 6.,
			new double[] { -216, -125, -64, -27,
					-8, -1, 0, 1, 8, 27, 64, 125, 216 });

	private static ArrayTabulatedFunction squareArray = new ArrayTabulatedFunction(-6., 6.,
			new double[] { 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36 });

	private static void toStringTest() {
		System.out.println("\ntoString() test:");
		System.out.print("Linked list: ");
		System.out.println(cubeLinkedList);
		System.out.print("Array: ");
		System.out.println(cubeArray);
	}

	private static void equalsTest() {
		System.out.println("\nequals() test:");
		System.out.printf("linked list cube == linked list square: %b\n", cubeLinkedList.equals(squareLinkedList));
		System.out.printf("linked list cube == linked list cube: %b\n", cubeLinkedListCopy.equals(cubeLinkedList));
		System.out.printf("array cube == linked list cube: %b\n", cubeArray.equals(cubeLinkedList));
		System.out.printf("array cube == linked list square: %b\n", cubeArray.equals(squareLinkedList));
	}

	private static void hashCodeTest() throws InappropriateFunctionPointException {
		System.out.println("\nhashCode() test:");
		System.out.println("Hash code of ...");
		System.out.printf("linked list cube: %d\n", cubeLinkedList.hashCode());
		System.out.printf("linked list cube copy: %d\n", cubeLinkedListCopy.hashCode());
		System.out.printf("linked list square: %d\n", squareLinkedList.hashCode());
		System.out.printf("array cube: %d\n", cubeArray.hashCode());
		System.out.printf("array square: %d\n", squareArray.hashCode());
		squareArray.setPoint(0, new FunctionPoint(-5.992, 35.904064));
		System.out.printf("array square after edit: %d\n", squareArray.hashCode());
	}

	private static void cloneTest() {
		System.out.println("\nclone() test:");
		System.out.printf("Initial object: %s\n", squareLinkedList.toString());
		LinkedListTabulatedFunction squareLinkedListClone = (LinkedListTabulatedFunction) squareLinkedList.clone();
		System.out.printf("Clone of object: %s\n", squareLinkedListClone.toString());
		squareLinkedList.deletePoint(0);
		System.out.printf("Initial object after modification: %s\n", squareLinkedList.toString());
		System.out.printf("Clone of object: %s\n", squareLinkedListClone.toString());

	}
}
