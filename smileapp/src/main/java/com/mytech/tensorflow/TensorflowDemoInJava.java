package com.mytech.tensorflow;

import java.util.List;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlowException;

public class TensorflowDemoInJava {

	public static void main(String[] args) {
		try (SavedModelBundle savedModelBundle = SavedModelBundle.load("tf_add_model","serve")) {

			try (Session session = savedModelBundle.session()) {
				System.out.println("\n SavedModelBundle :: " + savedModelBundle);
				Session.Runner runner = session.runner();
				runner.feed("x", Tensor.create(5));
				runner.feed("y", Tensor.create(10));
				List<Tensor<?>> tensors = runner.fetch("ans").run();
				System.out.println("\nAddition Operation Output is: " + tensors.get(0).intValue());
			}

		} catch (TensorFlowException ex) {
			ex.printStackTrace();
		}

		Graph graph = new Graph();
		Operation x = graph.opBuilder("Const", "x").setAttr("dtype", DataType.FLOAT)
				.setAttr("value", Tensor.create(3.0f)).build();

		Operation y = graph.opBuilder("Placeholder", "y").setAttr("dtype", DataType.FLOAT).build();

		Operation xy = graph.opBuilder("Mul", "xy").addInput(x.output(0)).addInput(y.output(0)).build();

		Session session = new Session(graph);
		System.out.println("------------------------------------------");
		Runner runnr = session.runner();
		runnr.feed("x", Tensor.create(10.0f));
		runnr.feed("y", Tensor.create(10.0f));
		Tensor<?> tensor = runnr.fetch("xy").run().get(0);
		// Tensor tensor = session.runner().fetch("xy").feed("x",
		// Tensor.create(5.0f)).feed("y", Tensor.create(2.0f)).run().get(0);
		System.out.println("Multiply Operation Output is :: " + tensor.floatValue());

		

	}
}
