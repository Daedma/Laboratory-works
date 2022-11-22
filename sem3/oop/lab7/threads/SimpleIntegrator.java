package threads;

public class SimpleIntegrator implements Runnable {
	Task task;

	public SimpleIntegrator(Task task) {
		this.task = task;
	}

	public void run() {
		for (int i = 0; i != task.getTaskCount(); ++i) {
			synchronized (task) {
				double result = task.integrate();
				System.out.printf("Result %f %f %f %f%n", task.getLeft(), task.getRight(), task.getStep(),
						result);
			}
		}
	}
}
