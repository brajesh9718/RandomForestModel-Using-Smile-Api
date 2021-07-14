package com.randomforest.train;

import java.io.File;
import java.util.Properties;

import com.model.dump.Pickle;

import smile.base.cart.SplitRule;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;

public class RandomForestForLeftTrainModel {

	public static void main(String[] args) {
		DataFrame data = null;
		try {
			data = Read.csv("rowfiles/left_train_rf.csv");
			System.out.println("======Random Forest Left Model========= ");
			Formula formula = Formula.lhs("V1");
			Properties params = new Properties();
			int ntrees = Integer.parseInt(params.getProperty("smile.random_forest.trees", "200"));
			int mtry = Integer.parseInt(params.getProperty("smile.random_forest.mtry", "5"));
			SplitRule rule = SplitRule.valueOf(params.getProperty("smile.random_forest.split_rule", "GINI"));
			int maxDepth = Integer.parseInt(params.getProperty("smile.random_forest.max_depth", "7"));
			int maxNodes = Integer.parseInt(params.getProperty("smile.random_forest.max_nodes", "30"));
			int nodeSize = Integer.parseInt(params.getProperty("smile.random_forest.node_size", "20"));
			// Train Model
			RandomForest model = RandomForest.fit(formula, data, ntrees, mtry, rule, maxDepth, maxNodes, nodeSize, 1, null, null);
			File file = new File("trainedModel\\left_trained.model");
			Pickle.dump(model, file);
			System.out.println("\nModel Trained and Dumped Successfully !");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
