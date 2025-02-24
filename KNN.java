import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.CoverTree;
import weka.core.neighboursearch.KDTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class KNN {

    public static void main(String[] args) throws Exception {
 
        if (args.length < 2) {
            System.out.println("ERROR: Please provide the path to the data file and the output file.");
            return;
        }

        String dataFilePath = args[0];
        String outputPath = args[1];

        Instances data = loadData(dataFilePath);
        if (data == null) return;

        // Replace missing values
        data = replaceMissingValues(data);

        int numInstances = data.numInstances();
        int[] distanceFunctions = {0, 1, 2}; // 0: LinearNNSearch, 1: CoverTree, 2: KDTree
        int[] distanceWeighting = {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE, IBk.WEIGHT_SIMILARITY};

        double bestFMeasure = 0;
        int bestK = 1;
        int bestD = 0;
        int bestW = IBk.WEIGHT_NONE;

        for (int k = 1; k <= numInstances; k++) {
            for (int d : distanceFunctions) {
                for (int w : distanceWeighting) {
                    IBk model = new IBk(k);
                    switch (d) {
                        case 0:
                            model.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());
                            break;
                        case 1:
                            model.setNearestNeighbourSearchAlgorithm(new CoverTree());
                            break;
                        case 2:
                            model.setNearestNeighbourSearchAlgorithm(new KDTree());
                            break;
                    }
                    model.setDistanceWeighting(new SelectedTag(w, IBk.TAGS_WEIGHTING));

                    model.buildClassifier(data);
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
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }
        Instances data = null;
        try {
            data = source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    private static Instances replaceMissingValues(Instances data) throws Exception {
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        return Filter.useFilter(data, replaceMissing);
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
            writer.write("      Metrika: " + (bestD == 0 ? "LinearNNSearch" : bestD == 1 ? "CoverTree" : "KDTree") + "\n");
            writer.write("      Distantziaren ponderazio faktorea: " + (bestW == IBk.WEIGHT_NONE ? "None" : bestW == IBk.WEIGHT_INVERSE ? "Inverse" : "Similarity") + "\n");
            writer.write("      F-Measure: " + bestFMeasure);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + outputPath + " fitxategian idatzi.");
        }
    }
}
