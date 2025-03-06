package LaboZaharrak;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class tryout {
    private static tryout instance;

    private tryout() {}

    public static tryout getObj() {
        if (instance == null) {
            instance = new tryout();
        }
        return instance;
    }

    public void execute() {
        try {
            DataSource ds = new DataSource("/home/ibai/Deskargak/heart-c.arff");
            Instances data = ds.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            Evaluation e = new Evaluation(data);
            System.out.println(e.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        tryout.getObj().execute();
    }
}
