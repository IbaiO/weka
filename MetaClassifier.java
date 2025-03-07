import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class MetaClassifier {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println("ERROR: Please provide the paths to the train file, test file, model file, and output predictions file.");
            return;
        }

        String trainFilePath = args[0];
        String testFilePath = args[1];
        String modelFilePath = args[2];
        String predictionsOutputPath = args[3];

        Instances trainData = loadData(trainFilePath);
        if (trainData == null) {
            System.out.println("ERROR: Failed to load train data.");
            return;
        }

        Instances testData = loadData(testFilePath);
        if (testData == null) {
            System.out.println("ERROR: Failed to load test data.");
            return;
        }

        System.out.println("Number of instances in train data: " + trainData.numInstances());
        System.out.println("Number of attributes in train data: " + trainData.numAttributes());
        System.out.println("Class attribute in train data: " + trainData.classAttribute());
        System.out.println("Number of classes in train data: " + trainData.numClasses());

        System.out.println("Number of instances in test data: " + testData.numInstances());
        System.out.println("Number of attributes in test data: " + testData.numAttributes());
        System.out.println("Class attribute in test data: " + testData.classAttribute());
        System.out.println("Number of classes in test data: " + testData.numClasses());

        // Train and save model using AttributeSelectedClassifier
        AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
        ASSearch search = new BestFirst();
        ASEvaluation eval = new CfsSubsetEval();
        asc.setSearch(search);
        asc.setEvaluator(eval);
        asc.setClassifier(new NaiveBayes());
        asc.buildClassifier(trainData);
        saveModel(asc, modelFilePath);

        // Load model and apply to test data
        AttributeSelectedClassifier loadedModel = loadModel(modelFilePath);
        applyModel(loadedModel, testData, predictionsOutputPath);

        System.out.println("Predictions completed successfully.");
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

        if (data == null) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Class index set to: " + data.classIndex());
        return data;
    }

    private static void saveModel(AttributeSelectedClassifier model, String filePath) throws Exception {
        try {
            SerializationHelper.write(filePath, model);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategian idatzi.");
        }
    }

    private static AttributeSelectedClassifier loadModel(String filePath) {
        try {
            return (AttributeSelectedClassifier) SerializationHelper.read(filePath);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }
    }

    private static void applyModel(AttributeSelectedClassifier model, Instances testData, String outputPath) throws Exception {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = model.classifyInstance(testData.instance(i));
                writer.write("Instance " + (i + 1) + ": Predicted class = " + testData.classAttribute().value((int) pred) + "\n");
            }
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + outputPath + " fitxategian idatzi.");
        }
    }
}
