import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EzZintzoa {
    public static void main(String[] args) throws Exception {
        Instances data = loadData("/home/ibai/Deskargak/heart-c.arff");
        if (data == null) return;
        String outputPath = "/home/ibai/Dokumentuak/weka/ez-zintzoa-results.txt";
        EzZintzoa.erabili_ez_zintzoa(data, outputPath, args);
    }

    public static void erabili_ez_zintzoa(Instances data, String outputPath, String[] args) throws Exception {
        NaiveBayes estimator = new NaiveBayes();
        estimator.buildClassifier(data);
        Evaluation evaluator = new Evaluation(data);
        evaluator.evaluateModel(estimator, data);

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

    private static Instances loadData(String filePath) {
        DataSource source = null;
        try {
            source = new DataSource(filePath);
        } catch (Exception e) {
            System.out.println("ERROR: File not found: " + filePath);
            return null;
        }
        Instances data = null;
        try {
            data = source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROR: Unable to read the file: " + filePath);
            return null;
        }
        
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        
        return data;
    }
}
