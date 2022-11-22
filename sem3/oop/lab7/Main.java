import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;

import functions.basic.Log;
import threads.Generator;
import threads.Integrator;
import threads.SimpleGenerator;
import threads.SimpleIntegrator;
import threads.Task;

public class Main {
	public static void main(String[] args) {
		// nonThread();
		try {
			// simpleThread();
			complicatedThreads();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void nonThread() {
		Task task = new threads.Task();
		task.setTaskCount(100);
		for (int i = 0; i != task.getTaskCount(); ++i) {
			task.setFunction(new Log(ThreadLocalRandom.current().nextDouble(Math.nextUp(1), 10)));
			task.setLeft(ThreadLocalRandom.current().nextDouble(Math.nextUp(0), 100));
			task.setRight(ThreadLocalRandom.current().nextDouble(100, 200));
			task.setStep(ThreadLocalRandom.current().nextDouble(Double.MIN_NORMAL, 1));
			System.out.printf("Source %f %f %f%n", task.getLeft(), task.getRight(), task.getStep());
			System.out.printf("Result %f %f %f %f%n", task.getLeft(), task.getRight(), task.getStep(),
					task.integrate());
		}
	}

	private static void simpleThread() throws InterruptedException {
		Task task = new Task();
		task.setTaskCount(100);
		SimpleGenerator simpleGenerator = new SimpleGenerator(task);
		SimpleIntegrator simpleIntegrator = new SimpleIntegrator(task);
		Thread generatorThread = new Thread(simpleGenerator);
		Thread integratorThread = new Thread(simpleIntegrator);
		generatorThread.start();
		integratorThread.start();
		generatorThread.join();
		integratorThread.join();
	}

	private static void complicatedThreads() throws InterruptedException {
		Task task = new Task();
		task.setTaskCount(100);
		Semaphore semaphore = new Semaphore(1, true);
		Generator generator = new Generator(task, semaphore);
		Integrator integrator = new Integrator(task, semaphore);
		// Thread generatorThread = new Thread(generator);
		// Thread integratorThread = new Thread(integrator);
		// generatorThread.start();
		// integratorThread.start();
		generator.start();
		integrator.start();
		Thread.currentThread().sleep(50);
		generator.interrupt();
		integrator.interrupt();
	}

}
