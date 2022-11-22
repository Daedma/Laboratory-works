package threads;

import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;

import functions.basic.Log;

public class Generator extends Thread {

	Task task;
	Semaphore semaphore;

	public Generator(Task task, Semaphore semaphore) {
		this.task = task;
		this.semaphore = semaphore;
	}

	public void run() {
		for (int i = 0; i != task.getTaskCount(); ++i) {
			semaphore.acquireUninterruptibly();
			task.setFunction(new Log(ThreadLocalRandom.current().nextDouble(Math.nextUp(1), 10)));
			task.setLeft(ThreadLocalRandom.current().nextDouble(Math.nextUp(0), 100));
			task.setRight(ThreadLocalRandom.current().nextDouble(100, 200));
			task.setStep(ThreadLocalRandom.current().nextDouble(Double.MIN_NORMAL, 1));
			System.out.printf("Source %f %f %f%n", task.getLeft(), task.getRight(), task.getStep());
			semaphore.release();
			if (isInterrupted())
				return;
		}
	};
}
