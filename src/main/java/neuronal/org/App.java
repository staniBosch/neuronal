package neuronal.org;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;

public class App 
{
	
	static {
		nu.pattern.OpenCV.loadShared();
		System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
	}
	
	
    public static void main( String[] args )
    {
    	int hiddenLayerSize = 4;
		//
		int row = 0;
		int col = 0;

		float inputTrainingDataArray[] = {0.0f,0.0f,
				0.0f,1.0f,
				1.0f,0.0f,
				1.0f,1.0f};
				
		// allocate Mat before calling put		
		Mat inputTrainingData = new Mat(4,2,CvType.CV_32F);
		inputTrainingData.put(row, col, inputTrainingDataArray);
		System.out.println(inputTrainingData.dump());
		
		float outputTrainDataArray[] = {0.0f, 1.0f, 1.0f, 0.0f};
		
		Mat outputTrainingData = new Mat(4,1,CvType.CV_32F);
		outputTrainingData.put(row, col, outputTrainDataArray);
		System.out.println(outputTrainingData.dump());
		
		ANN_MLP mlp = ANN_MLP.create();
		Mat layersSize = new Mat(3,1,CvType.CV_16U);
		layersSize.row(0).setTo(new Scalar(inputTrainingData.cols()));
		layersSize.row(1).setTo(new Scalar(hiddenLayerSize));
		layersSize.row(2).setTo(new Scalar(outputTrainingData.cols()));
		System.out.println(layersSize.dump());
		
		mlp.setLayerSizes(layersSize);
		mlp.setActivationFunction(ANN_MLP.SIGMOID_SYM, 0.0,0.0);
		
		TermCriteria termCrit = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
				1000,0.00001);
		mlp.setTermCriteria(termCrit);
		
		mlp.setTrainMethod(ANN_MLP.BACKPROP);
		
		mlp.train(inputTrainingData, Ml.ROW_SAMPLE, outputTrainingData);
		System.out.println("Ergebnis (Boolean) von isTrained: " + mlp.isTrained());
		System.out.println("Ergebnis (Boolean) von is Classifier: "+mlp.isClassifier());
		
		for (int i=0;i<inputTrainingData.rows();i++){
			Mat sample = new Mat(1, inputTrainingData.cols(),CvType.CV_32F);
			sample.put(0, 0, new float[]{(float)(inputTrainingData.get(i,0)[0]),
					(float)(inputTrainingData.get(i, 1)[0])});
			System.out.println(sample.dump());
			Mat results = new Mat();
			
			mlp.predict(sample, results,0);
			System.out.println("Result" + i +": " +results.dump());
		}
    }
}
