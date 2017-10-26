package lhrotk.neuralnet;
/**
 * 
 * @author lhrotk
 *
 */
public class TrainingData {
	private double[]	inputVector;
	private double[] 	outputVector;
	public TrainingData(double[] input, double[] output) {
		this.inputVector = input;
		this.outputVector = output;
	}
	public double[] getInputVector() {
		return inputVector;
	}
	public void setInputVector(double[] inputVector) {
		this.inputVector = inputVector;
	}
	public double[] getOutputVector() {
		return outputVector;
	}
	public void setOutputVector(double[] outputVector) {
		this.outputVector = outputVector;
	}
}
