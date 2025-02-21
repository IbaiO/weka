import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class KNN {

    public static void main(String[] args) throws Exception {
/* 
        if (args.length < 2) {
            System.out.println("ERROR: Please provide the path to the data file and the output file.");
            return;
        }

        String dataFilePath = args[0];
        String outputPath = args[1];
        */

        String dataFilePath = "/home/ibai/Deskargak/heart-c.arff";
        String outputPath = "/home/ibai/Dokumentuak/weka/knn-emiatza.txt";

        Instances data = loadData(dataFilePath);
        if (data == null) return;

        int[] kValues = {1, 3, 5, 7, 9, 11, 13, 15};
        int[] distanceWeighting = {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE, IBk.WEIGHT_SIMILARITY};
        int[] windowSizes = {0, 50, 100, 150, 200};

        double bestFMeasure = 0;
        int bestK = 1;
        int bestD = IBk.WEIGHT_NONE;
        int bestW = 0;

        for (int k : kValues) {
            for (int d : distanceWeighting) {
                for (int w : windowSizes) {
                    IBk model = new IBk(k);
                    model.setDistanceWeighting(new SelectedTag(d, IBk.TAGS_WEIGHTING));
                    model.setWindowSize(w);

                    Evaluation evaluation = evaluateModel(data, model);
                    double fMeasure = evaluation.weightedFMeasure();

                    if (fMeasure > bestFMeasure) {
                        bestFMeasure = fMeasure;
                        bestK = k;
                        bestD = d;
                        bestW = w;
                    }
                }
            }
        }

        saveResults(outputPath, bestK, bestD, bestW, bestFMeasure);
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

    private static Evaluation evaluateModel(Instances data, IBk model) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(model, data, 10, new Random(1));
        return evaluation;
    }

    private static void saveResults(String outputPath, int bestK, int bestD, int bestW, double bestFMeasure) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            writer.write("Parametrorik hoberenak:\n");
            writer.write("      Auzokide kopurua: " + bestK + "\n");
            writer.write("      M etrika: " + (bestD == IBk.WEIGHT_NONE ? "None" : bestD == IBk.WEIGHT_INVERSE ? "Inverse" : "Similarity") + "\n");
            writer.write("      Distantziaren ponderazio faktorea: " + bestW + "\n");
            writer.write("      F-Measure hoberena: " + bestFMeasure + "\n");
        } catch (IOException e) {
            System.out.println("ERROR: Unable to write to the file: " + outputPath);
        }
    }
}
