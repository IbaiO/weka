import java.io.IOException;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

public class FSSyNB {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("ERROR: Please provide the paths to the train file and the output model file.");
            return;
        }

        String trainFilePath = args[0];
        String modelOutputPath = args[1];

        Instances trainData = loadData(trainFilePath);
        if (trainData == null) {
            System.out.println("ERROR: Failed to load train data.");
            return;
        }

        System.out.println("Number of instances: " + trainData.numInstances());
        System.out.println("Number of attributes: " + trainData.numAttributes());
        System.out.println("Class attribute: " + trainData.classAttribute());
        System.out.println("Number of classes: " + trainData.numClasses());

        Instances selectedTrainData = selectAttributes(trainData);

        System.out.println("Number of attributes before selection: " + trainData.numAttributes());
        System.out.println("Number of attributes after selection: " + selectedTrainData.numAttributes());

        NaiveBayes nb = trainModel(selectedTrainData);

        saveModel(nb, modelOutputPath);

        System.out.println("Naive Bayes model saved successfully.");
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

    private static Instances selectAttributes(Instances data) throws Exception {
        AttributeSelection attributeSelection = new AttributeSelection();
        ASSearch search = new BestFirst();
        CfsSubsetEval eval = new CfsSubsetEval();
        attributeSelection.setSearch(search);
        attributeSelection.setEvaluator(eval);
        attributeSelection.SelectAttributes(data);
        return attributeSelection.reduceDimensionality(data);
    }

    private static NaiveBayes trainModel(Instances data) throws Exception {
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);
        return nb;
    }

    private static void saveModel(NaiveBayes model, String filePath) throws Exception {
        try {
            SerializationHelper.write(filePath, model);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategian idatzi.");
        }
    }
}
