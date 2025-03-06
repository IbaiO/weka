package LaboZaharrak;
// Java Program to Illustrate Usage of Weka API

// Importing required classes
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

// Heart disease
public class GFG {

	public static void main(String args[]) {
		try {

			
			J48 j48Classifier = new J48();	// Create J48 classifier by creating object of J48 class

			String heartDiseaseDataset = "/home/ibai/Deskargak/heart-c.arff";	// Dataset path

			BufferedReader bufferedReader = new BufferedReader(new FileReader(heartDiseaseDataset));	// Creating bufferedreader to read the dataset

			Instances datasetInstances = new Instances(bufferedReader);	// Create dataset instances
			
			datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);	// Set Target Class
			
			Evaluation evaluation = new Evaluation(datasetInstances);	// Evaluating by creating object of Evaluation class
			evaluation.crossValidateModel(j48Classifier, datasetInstances, 10, new Random(1)); // Cross Validate Model with 10 folds

			System.out.println(evaluation.toSummaryString("\nResults", false));
		}

		catch (Exception e) {
			// Print message on the console
			System.out.println("Errore bat gertatu da! \n" + e.getMessage());
		}
	}
}
