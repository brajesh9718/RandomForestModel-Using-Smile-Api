package com.randomforest.test;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.model.dump.Pickle;
import com.randomforest.util.RandomForestUtil;

import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.io.Read;

public class RandomForestForRightTestModel {

	private static final String inputModel = "trainedModel/right_trained.model";
	private static final String inputTest = "rowfiles/scaled_right_test.csv";

	public static Map<String, Object> getRightRFPrediction(String inputTest, String inputModel) {
		Map<String, Object> mapVotesProb = null;
		try {
			// Load and Predict Model...
			System.out.println("Inside getRightRFPrediction()======");
			File leftModelFile = new File(inputModel);
			RandomForest loadModel = (RandomForest) Pickle.load(leftModelFile);
			DataFrame testData = Read.csv(inputTest);
			int n = testData.size();
			Tuple xj = null;
			double[] posteriori = new double[n];
			Map<String, double[]> mapposteriori = new HashMap<String, double[]>();
			Map<String, Integer> mapVotes = new HashMap<String, Integer>();
			Integer[] votes = new Integer[n];
			int indx = 0;
			for (int j = 0; j < n; j++) {
				xj = testData.get(j);
				// for vote....
				int vote = loadModel.vote(xj, new double[4]);
				// for posteriori
				posteriori = RandomForestUtil.rf_vote(xj, new double[4], loadModel);
				String probindx = "prob_" + indx;
				mapposteriori.put(probindx, posteriori);
				mapVotes.put(probindx, vote);
				votes[j] = vote;
				indx++;
			}
			mapVotesProb = new HashMap<String, Object>();
			mapVotesProb.put("votes", mapVotes);
			mapVotesProb.put("posteriori", mapposteriori);

		} catch (Exception e) {
			e.printStackTrace();
		}
		return mapVotesProb;

	}

	public static void main(String[] args) {
		try {
			// Load and Predict Model...
			File leftModelFile = new File(inputModel);
			RandomForest loadModel = (RandomForest) Pickle.load(leftModelFile);
			DataFrame testData = Read.csv(inputTest);
			int n = testData.size();
			Tuple xj = null;
			double[] posteriori = new double[n];
			Map<String, double[]> mapposteriori = new HashMap<String, double[]>();
			Map<String, Integer> mapVotes = new HashMap<String, Integer>();
			Integer[] votes = new Integer[n];
			int indx = 0;
			for (int j = 0; j < n; j++) {
				xj = testData.get(j);
				// for vote....
				int vote = loadModel.vote(xj, new double[4]);
				// for posteriori
				posteriori = RandomForestUtil.rf_vote(xj, new double[4], loadModel);
				String probindx = "prob_" + indx;
				mapposteriori.put(probindx, posteriori);
				mapVotes.put(probindx, vote);
				votes[j] = vote;
				indx++;
			}
			Map<String, Object> mapVotesProb = new HashMap<String, Object>();
			mapVotesProb.put("votes", mapVotes);
			mapVotesProb.put("posteriori", mapposteriori);
			
			Map<String, Integer> votesRes = (Map<String, Integer>) mapVotesProb.get("votes");			
			votesRes.forEach((k, v) -> System.out.println("Key :" + k + "   Value : " + v));
			
			/*for(Map.Entry<String, Integer> prob: votesRes.entrySet()) {
				System.out.println("----------------");
				Integer voteOut =  votesRes.get(prob.getKey());
				System.out.println(voteOut);
				
			}*/
			
			Map<String, double[]> probRes = (Map<String, double[]>) mapVotesProb.get("posteriori");			
			for(Map.Entry<String, double[]> prob: probRes.entrySet()) {
				//System.out.println("----------------");
				double[] proOut =  probRes.get(prob.getKey());
				Arrays.stream(proOut).forEach((proOut1) -> System.out.println(proOut1));
				
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
}
