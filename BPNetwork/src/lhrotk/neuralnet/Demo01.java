package lhrotk.neuralnet;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import lhrotk.neuralnet.exception.InputFormatException;
import lhrotk.test.TestPlan;

public class Demo01 {

	public static void main(String[] args) throws IOException, InputFormatException {
		BinaryNeuralNet bNet = new BinaryNeuralNet(2, 4, 1, 0.2, 0, 0, 0);
		TrainingData data1 = new TrainingData(new double[] {1,0}, new double[] {1});
		TrainingData data2 = new TrainingData(new double[] {0,1}, new double[] {1});
		TrainingData data3 = new TrainingData(new double[] {1,1}, new double[] {0});
		TrainingData data4 = new TrainingData(new double[] {0,0}, new double[] {0});
		List<TrainingData> trainingSet = new ArrayList<TrainingData>();
		trainingSet.add(data1);
		trainingSet.add(data2);
		trainingSet.add(data3);
		trainingSet.add(data4);
		//bNet.setInitial();
		bNet.trainingPlan(trainingSet, 0);
		System.out.println(bNet.getTotalError(trainingSet));
		File argFile = new File("demo01w.txt");
		bNet.save(argFile);
		BinaryNeuralNet bNet2 = new BinaryNeuralNet(1,1,1,0,0,0, 0);
		bNet2.load("demo01w.txt");
		System.out.println(bNet2.getTotalError(trainingSet));
	}

}
