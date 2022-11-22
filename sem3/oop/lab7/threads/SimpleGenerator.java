package threads;

import java.util.concurrent.ThreadLocalRandom;
import functions.basic.Log;

public class SimpleGenerator implements Runnable {
	private Task task;

	public SimpleGenerator(Task task) {
		this.task = task;
	}

	public void run() {
		for (int i = 0; i != task.getTaskCount(); ++i) {
			synchronized (task) {
				task.setFunction(new Log(ThreadLocalRandom.current().nextDouble(Math.nextUp(1), 10)));
				task.setLeft(ThreadLocalRandom.current().nextDouble(Math.nextUp(0), 100));
				task.setRight(ThreadLocalRandom.current().nextDouble(100, 200));
				task.setStep(ThreadLocalRandom.current().nextDouble(Double.MIN_NORMAL, 1));
			}
			System.out.printf("Source %f %f %f%n", task.getLeft(), task.getRight(), task.getStep());
		}
	}

}
