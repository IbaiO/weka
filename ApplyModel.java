import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class ApplyModel {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("ERROR: Please provide the paths to the test file, model file, and output predictions file.");
            return;
        }

        String testFilePath = args[0];
        String modelFilePath = args[1];
        String predictionsOutputPath = args[2];

        Instances testData = loadData(testFilePath);
        if (testData == null) {
            System.out.println("ERROR: Failed to load test data.");
            return;
        }

        System.out.println("Number of instances: " + testData.numInstances());
        System.out.println("Number of attributes: " + testData.numAttributes());
        System.out.println("Class attribute: " + testData.classAttribute());
        System.out.println("Number of classes: " + testData.numClasses());

        NaiveBayes nb = loadModel(modelFilePath);
        if (nb == null) {
            System.out.println("ERROR: Failed to load model.");
            return;
        }

        Instances testDataNoClass = removeClassAttribute(testData);
        if (testDataNoClass == null) {
            System.out.println("ERROR: Failed to remove class attribute from test data.");
            return;
        }

        // Ensure class index is set correctly after removing class attribute
        if (testDataNoClass.classIndex() == -1) {
            testDataNoClass.setClassIndex(testDataNoClass.numAttributes() - 1);
        }
        System.out.println("Class index set to: " + testDataNoClass.classIndex());

        applyModel(nb, testData, testDataNoClass, predictionsOutputPath);

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

    private static NaiveBayes loadModel(String filePath) {
        try {
            return (NaiveBayes) SerializationHelper.read(filePath);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }
    }

    private static Instances removeClassAttribute(Instances data) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.classIndex() + 1));
        remove.setInputFormat(data);
        return Filter.useFilter(data, remove);
    }

    private static void applyModel(NaiveBayes model, Instances originalData, Instances dataNoClass, String outputPath) throws Exception {
        if (model == null || originalData == null || dataNoClass == null) {
            System.out.println("ERROR: Model, original data, or data without class is null.");
            return;
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            for (int i = 0; i < dataNoClass.numInstances(); i++) {
                double pred = model.classifyInstance(dataNoClass.instance(i));
                writer.write("Instance " + (i + 1) + ": Predicted class = " + originalData.classAttribute().value((int) pred) + "\n");
            }
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + outputPath + " fitxategian idatzi.");
        }
    }
}