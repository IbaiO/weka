import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class Stratified70percentSplit {

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("ERROR: Please provide the paths to the data file, train output file, and test output file.");
            return;
        }

        String dataFilePath = args[0];
        String trainOutputPath = args[1];
        String testOutputPath = args[2];

        Instances data = loadData(dataFilePath);
        if (data == null) {
            System.out.println("ERROR: Failed to load data.");
            return;
        }

        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute());
        System.out.println("Number of classes: " + data.numClasses());

        Instances trainData = createTrainData(data);
        Instances testData = createTestData(data);

        saveData(trainData, trainOutputPath);
        saveData(testData, testOutputPath);

        System.out.println("Train and test data saved successfully.");
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

    private static Instances createTrainData(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0);
        resample.setNoReplacement(true);
        resample.setSampleSizePercent(70);
        resample.setInputFormat(data);
        return Filter.useFilter(data, resample);
    }

    private static Instances createTestData(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setNoReplacement(true);
        resample.setSampleSizePercent(30);
        resample.setInvertSelection(true);
        resample.setInputFormat(data);
        Instances testData = Filter.useFilter(data, resample);

        for (int i = 0; i < testData.numInstances(); i++) {
            testData.instance(i).setClassMissing();
        }
        return testData;
    }

    private static void saveData(Instances data, String filePath) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(filePath));
            saver.writeBatch();
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategian idatzi.");
        }
    }
}
