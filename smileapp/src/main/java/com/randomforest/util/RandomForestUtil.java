package com.randomforest.util;

import java.util.Arrays;

import smile.classification.RandomForest;
import smile.data.Tuple;
import smile.data.formula.Formula;

public class RandomForestUtil {

	public static double[] rf_vote(Tuple x, double[] posteriori, RandomForest loadModel) {
		if (posteriori.length != 4) {
			throw new IllegalArgumentException(
					String.format("Invalid posteriori vector size: %d, expected: %d", posteriori.length, 4));
		}
		Formula formula = loadModel.formula();
		Tuple xt = formula.x(x);
		Arrays.fill(posteriori, 0.0);
		for (RandomForest.Model model : loadModel.models()) {
			// System.out.println("model.tree.predict(xt) :: "+ model.tree.predict(xt));
			posteriori[model.tree.predict(xt)]++;
		}
		double postsum = 0;
		for (int i = 0; i < posteriori.length; i++) {
			postsum = postsum + posteriori[i];

		}
		for (int i = 0; i < posteriori.length; i++) {
			posteriori[i] = posteriori[i] / postsum;
		}
		 //System.out.println("posteriori::");
		 //Arrays.stream(posteriori).forEach(System.out::println);
		return posteriori;
	}
}
