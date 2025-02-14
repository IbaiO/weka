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
            //writer.write("Execution Date: " + new Date() + "\n");
            writer.write("Argumentuak: " + String.join(", ", args) + "\n\n");
            writer.write("Nahasmen matrizea:\n");
            double[][] confMatrix = evaluator.confusionMatrix();
            for (double[] row : confMatrix) {
                for (double value : row) {
                    writer.write(value + "\t");
                }
                writer.write("\n");
            }
            writer.write("\n");

            writer.write("Ebaluazio metrikak:\n");
            for (int i = 0; i < data.numClasses(); i++) {
                writer.write("Klasea: " + i + " --> " + evaluator.precision(i) + "\n");
            }

            // Add formulas for Precision, Recall, Accuracy, and F-measure
            writer.write("Metrics:\n");
            for (int i = 0; i < data.numClasses(); i++) {
                double tp = evaluator.numTruePositives(i);
                double fp = evaluator.numFalsePositives(i);
                double fn = evaluator.numFalseNegatives(i);
                double tn = evaluator.numTrueNegatives(i);

                double precision = tp / (tp + fp);
                double recall = tp / (tp + fn);
                double accuracy = (tp + tn) / (tp + tn + fp + fn);
                double fMeasure = (2 * precision * recall) / (precision + recall);

                writer.write("Class " + i + ":\n");
                writer.write("Precision: " + precision + "\n");
                writer.write("Recall: " + recall + "\n");
                writer.write("Accuracy: " + accuracy + "\n");
                writer.write("F-Measure: " + fMeasure + "\n\n");
            }

            /* 
            writer.write("Weighted Avg: " + evaluator.weightedPrecision() + "\n\n");

            writer.write("Ebaluazio emaitzak:\n");
            writer.write("Zuzen klasifikatutako emaitzak: " + evaluator.pctCorrect() + "\n");
            writer.write("Oker klasifikatutako emaitzak: " + evaluator.pctIncorrect() + "\n");
            writer.write("Kappa Statistic: " + evaluator.kappa() + "\n");
            writer.write("Batazbesteko errore absolutua: " + evaluator.meanAbsoluteError() + "\n");
            writer.write("Root Mean Squared Error: " + evaluator.rootMeanSquaredError() + "\n");
            writer.write("Errore absolutu erlatiboa: " + evaluator.relativeAbsoluteError() + "\n");
            writer.write("Root Relative Squared Error: " + evaluator.rootRelativeSquaredError() + "\n");
            */
        } catch (IOException e) {
            System.out.println("ERROR: Fitxategia ezin izan da irakurri: " + outputPath);
        }
    }

    private static Instances loadData(String filePath) {
        DataSource source = null;
        try {
            source = new DataSource(filePath);
        } catch (Exception e) {
            System.out.println("ERROR: Fitxategia ez da aurkitu: " + filePath);
            return null;
        }
        Instances data = null;
        try {
            data = source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROR: Fitxategia ezin izan da irakurri: " + filePath);
            return null;
        }
        
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        
        return data;
    }
}
