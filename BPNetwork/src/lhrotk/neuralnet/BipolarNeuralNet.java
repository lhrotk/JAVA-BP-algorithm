package lhrotk.neuralnet;

import java.util.ArrayList;
import java.util.List;

import lhrotk.neuralnet.exception.InputFormatException;
import lhrotk.test.TestPlan;

public class BipolarNeuralNet extends AbstractNeuralNet{

	public BipolarNeuralNet(int argNumInputs, int argNumHidden, int argNumOutputs, double argLearningRate,
			double argMomentumTerm, double argA, double argB) throws InputFormatException {
		super(argNumInputs, argNumHidden, argNumOutputs, argLearningRate, argMomentumTerm, argA, argB);
	}

	public static void main(String[] args) throws InputFormatException {
		BipolarNeuralNet bNet = new BipolarNeuralNet(2, 4, 1, 0.2, 0.9, 0, 0);
		TrainingData data1 = new TrainingData(new double[] {1,-1}, new double[] {1});
		TrainingData data2 = new TrainingData(new double[] {-1,1}, new double[] {1});
		TrainingData data3 = new TrainingData(new double[] {1,1}, new double[] {-1});
		TrainingData data4 = new TrainingData(new double[] {-1,-1}, new double[] {-1});
		List<TrainingData> trainingSet = new ArrayList<TrainingData>();
		trainingSet.add(data1);
		trainingSet.add(data2);
		trainingSet.add(data3);
		trainingSet.add(data4);
		TestPlan testPlan = new TestPlan();
		for(int i=0; i<1000; i++) {
			bNet.initializeWeights();
			int result = bNet.trainingPlan(trainingSet, 1000);
			if(result>=0){
				testPlan.addNewData(result);
			}else {
				testPlan.addNewNonConverge();
			}
		}
		System.out.println("max:" + testPlan.getMax());
		System.out.println("min:" + testPlan.getMin());
		System.out.println("avg:" + testPlan.getAverage());
		System.out.println("non-converge rate:" + testPlan.getNonConvergeRate());

	}

	@Override
	double activFunction(double x) {
		// TODO Auto-generated method stub
		return -1+2*this.sigmoid(x);
	}

	@Override
	double diffActivFun(double fX) {
		// TODO Auto-generated method stub
		return 0.5*(1-fX*fX);
	}
	
	public void setInitial() {
		this.weightLayer1[0] = new double[]{0.1970, 0.3191, -0.1448, 0.3594};
		this.weightLayer1[1] = new double[]{0.3099, 0.1904, -0.0347, -0.4861};
		this.weightLayer1[2] = new double[]{-0.3378, 0.2771, 0.2859, -0.3329};
		this.weightLayer2[0] = new double[]{0.4919};
		this.weightLayer2[1] = new double[]{-0.2913};
		this.weightLayer2[2] = new double[]{-0.3979};
		this.weightLayer2[3] = new double[]{0.3581};
		this.weightLayer2[4] = new double[]{-0.1401};
	}

}
