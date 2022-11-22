package threads;

import java.util.concurrent.Semaphore;

public class Integrator extends Thread {

	Task task;
	Semaphore semaphore;

	public Integrator(Task task, Semaphore semaphore) {
		this.task = task;
		this.semaphore = semaphore;
	}

	public void run() {
		for (int i = 0; i != task.getTaskCount(); ++i) {
			semaphore.acquireUninterruptibly();
			double result = task.integrate();
			System.out.printf("Result %f %f %f %f%n", task.getLeft(), task.getRight(), task.getStep(),
					result);
			semaphore.release();
			if (isInterrupted())
				return;
		}
	};
}
