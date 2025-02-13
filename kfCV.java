import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class kfCV {
    public static void perform5FoldCV(Instances data, String outputPath, String[] args) throws Exception {
        NaiveBayes estimator = new NaiveBayes();
        Evaluation evaluator = new Evaluation(data);
        evaluator.crossValidateModel(estimator, data, 5, new Random(1));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            writer.write("Execution Date: " + new Date() + "\n");
            writer.write("Execution Arguments: " + String.join(", ", args) + "\n\n");
            writer.write("Confusion Matrix:\n");
            double[][] confMatrix = evaluator.confusionMatrix();
            for (double[] row : confMatrix) {
                for (double value : row) {
                    writer.write(value + "\t");
                }
                writer.write("\n");
            }
            writer.write("\n");

            writer.write("Precision Metrics:\n");
            for (int i = 0; i < data.numClasses(); i++) {
                writer.write("Class " + i + ": " + evaluator.precision(i) + "\n");
            }
            writer.write("Weighted Avg: " + evaluator.weightedPrecision() + "\n\n");

            writer.write("Evaluation Results:\n");
            writer.write("Correctly Classified Instances: " + evaluator.pctCorrect() + "\n");
            writer.write("Incorrectly Classified Instances: " + evaluator.pctIncorrect() + "\n");
            writer.write("Kappa Statistic: " + evaluator.kappa() + "\n");
            writer.write("Mean Absolute Error: " + evaluator.meanAbsoluteError() + "\n");
            writer.write("Root Mean Squared Error: " + evaluator.rootMeanSquaredError() + "\n");
            writer.write("Relative Absolute Error: " + evaluator.relativeAbsoluteError() + "\n");
            writer.write("Root Relative Squared Error: " + evaluator.rootRelativeSquaredError() + "\n");
        } catch (IOException e) {
            System.out.println("ERROR: Unable to write to the file: " + outputPath);
        }
    }
}
