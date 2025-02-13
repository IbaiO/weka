import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Date;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ModelSortu {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("ERROR: Please provide the path to the data file, the model storage path, and the quality estimation file path as arguments.");
            return;
        }

        //String dataFilePath = args[0];
        //String modelPath = args[1];
        //String qualityEstimationPath = args[2];
        String dataFilePath = "/home/ibai/Deskargak/heart-c.arff";
        String modelPath = "/home/ibai/Dokumentuak/weka/naive-bayes-model.model";
        String qualityEstimationPath = "/home/ibai/Dokumentuak/weka/naive-bayes-results.txt";

        Instances data = loadData(dataFilePath);
        if (data == null) return;

        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(data);

        // Save the model
        saveModel(model, modelPath);

        // Evaluate the model using k-fold cross-validation
        Evaluation kfEvaluation = evaluateKFold(data, 10);

        // Evaluate the model using hold-out (70% training, 30% testing)
        Evaluation hoEvaluation = evaluateHoldOut(data, 0.7);

        // Save the evaluation results
        saveEvaluationResults(kfEvaluation, hoEvaluation, qualityEstimationPath, args, data.numClasses());
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

    private static void saveModel(NaiveBayes model, String modelPath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
            oos.writeObject(model);
        } catch (IOException e) {
            System.out.println("ERROR: Unable to save the model: " + modelPath);
        }
    }

    private static Evaluation evaluateKFold(Instances data, int k) throws Exception {
        NaiveBayes model = new NaiveBayes();
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(model, data, k, new Random(1));
        return evaluation;
    }

    private static Evaluation evaluateHoldOut(Instances data, double trainSize) throws Exception {
        data.randomize(new Random(1));
        int trainSizeInt = (int) Math.round(data.numInstances() * trainSize);
        int testSizeInt = data.numInstances() - trainSizeInt;
        Instances train = new Instances(data, 0, trainSizeInt);
        Instances test = new Instances(data, trainSizeInt, testSizeInt);

        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(train);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(model, test);
        return evaluation;
    }

    private static void saveEvaluationResults(Evaluation kfEvaluation, Evaluation hoEvaluation, String qualityEstimationPath, String[] args, int numClasses) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(qualityEstimationPath))) {
            writer.write("Execution Date: " + new Date() + "\n");
            writer.write("Execution Arguments: " + String.join(", ", args) + "\n\n");

            writer.write("K-Fold Cross-Validation Results:\n");
            writeEvaluationResults(writer, kfEvaluation, numClasses);

            writer.write("\nHold-Out (70% Training, 30% Testing) Results:\n");
            writeEvaluationResults(writer, hoEvaluation, numClasses);
        } catch (IOException e) {
            System.out.println("ERROR: Unable to write to the file: " + qualityEstimationPath);
        }
    }

    private static void writeEvaluationResults(BufferedWriter writer, Evaluation evaluation, int numClasses) throws IOException {
        writer.write("Confusion Matrix:\n");
        double[][] confMatrix = evaluation.confusionMatrix();
        for (double[] row : confMatrix) {
            for (double value : row) {
                writer.write(value + "\t");
            }
            writer.write("\n");
        }
        writer.write("\n");

        writer.write("Precision Metrics:\n");
        for (int i = 0; i < numClasses; i++) {
            writer.write("Class " + i + ": " + evaluation.precision(i) + "\n");
        }
        writer.write("Weighted Avg: " + evaluation.weightedPrecision() + "\n\n");

        writer.write("Evaluation Results:\n");
        writer.write("Correctly Classified Instances: " + evaluation.pctCorrect() + "\n");
        writer.write("Incorrectly Classified Instances: " + evaluation.pctIncorrect() + "\n");
        writer.write("Kappa Statistic: " + evaluation.kappa() + "\n");
        writer.write("Mean Absolute Error: " + evaluation.meanAbsoluteError() + "\n");
        writer.write("Root Mean Squared Error: " + evaluation.rootMeanSquaredError() + "\n");
        try {
            writer.write("Relative Absolute Error: " + evaluation.relativeAbsoluteError() + "\n");
        } catch (Exception e) {
            writer.write("Relative Absolute Error: Error\n");
        }
        writer.write("Root Relative Squared Error: " + evaluation.rootRelativeSquaredError() + "\n");
    }
}
