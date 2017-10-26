package lhrotk.neuralnet;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import lhrotk.neuralnet.exception.*;

/**
 * 
 * @author lhrotk
 *
 */
abstract class AbstractNeuralNet implements NeuralNetInterface {
	final double SIGMOID_LOWER_BOUND;
	final double SIGMOID_UPPER_BOUND;
	final double LEARNING_RATE;
	final double MOMENTUM_TERM;
	final int DIM_OF_INPUT;
	final int DIM_OF_HIDDEN;
	final int DIM_OF_OUTPUT;

	final int INPUT_TO_HIDDEN = 0;
	final int HIDDEN_TO_OUTPUT = 1;

	private double acceptableError = 0.05;

	protected double[][] weightLayer1;
	protected double[][] weightLayer2;

	private double[][] weightChange1;
	private double[][] weightChange2;

	private double[] hiddenNeurons;
	private double[] outputVector;

	/**
	 * 
	 * @param argNumInputs
	 *            dimension of input vector;
	 * @param argNumHidden
	 *            dimension of hidder layer;
	 * @param argNumOutputs
	 *            dimension of output vector;
	 * @param argLearningRate
	 *            learing rate ro;
	 * @param argMomentumTerm
	 *            momentum alpha;
	 * @param argA
	 *            upper bound in customsigmoid;
	 * @param argB
	 *            upper bound in cuustom sigmoid;
	 * @return
	 */

	public AbstractNeuralNet(int argNumInputs, int argNumHidden, int argNumOutputs, double argLearningRate,
			double argMomentumTerm, double argA, double argB) throws InputFormatException {
		this.SIGMOID_LOWER_BOUND = argA;
		this.SIGMOID_UPPER_BOUND = argB;
		this.DIM_OF_INPUT = argNumInputs;
		this.DIM_OF_HIDDEN = argNumHidden;
		this.DIM_OF_OUTPUT = argNumOutputs;
		this.LEARNING_RATE = argLearningRate;
		this.MOMENTUM_TERM = argMomentumTerm;
		if (this.DIM_OF_HIDDEN < 1 || this.DIM_OF_INPUT < 1 || this.DIM_OF_OUTPUT < 1) {
			throw new InputFormatException("Layer Dimension Initialization Exception");
		}
		// consider bias terms
		this.weightLayer1 = new double[this.DIM_OF_INPUT + 1][this.DIM_OF_HIDDEN];
		this.weightLayer2 = new double[this.DIM_OF_HIDDEN + 1][this.DIM_OF_OUTPUT];
		this.weightChange1 = new double[this.DIM_OF_INPUT + 1][this.DIM_OF_HIDDEN];
		this.weightChange2 = new double[this.DIM_OF_HIDDEN + 1][this.DIM_OF_OUTPUT];
		this.hiddenNeurons = new double[this.DIM_OF_HIDDEN + 1];
		this.hiddenNeurons[this.hiddenNeurons.length - 1] = NeuralNetInterface.BIAS;
		this.outputVector = new double[this.DIM_OF_OUTPUT];
		// initialize weights
		this.initializeWeights();
	}

	@Override
	public double[] outputFor(double[] inputVector) throws InputFormatException {
		if (inputVector.length != this.DIM_OF_INPUT) {
			throw new InputFormatException("input dimension mismatch");
		}
		// from input to hidden
		for (int j = 0; j < this.DIM_OF_HIDDEN; j++) {
			this.hiddenNeurons[j] = 0;
			for (int i = 0; i < this.DIM_OF_INPUT; i++) {
				this.hiddenNeurons[j] += inputVector[i] * weightLayer1[i][j];
			}
			this.hiddenNeurons[j] += NeuralNetInterface.BIAS * weightLayer1[this.DIM_OF_INPUT][j];
			this.hiddenNeurons[j] = this.activFunction(hiddenNeurons[j]);
		}
		// from hidden to output
		for (int j = 0; j < this.DIM_OF_OUTPUT; j++) {
			this.outputVector[j] = 0;
			for (int i = 0; i < this.DIM_OF_HIDDEN + 1; i++) {
				this.outputVector[j] += this.hiddenNeurons[i] * weightLayer2[i][j];
			}
			this.outputVector[j] = this.activFunction(this.outputVector[j]);
		}
		return this.outputVector;
	}

	@Override
	public double train(double[] inputVector, double[] argValue) throws InputFormatException {
		if (inputVector.length != this.DIM_OF_INPUT || argValue.length != this.DIM_OF_OUTPUT) {
			throw new InputFormatException("input dimension mismatch");
		}
		// namely yj
		this.outputFor(inputVector);
		// do back propagation: update weight from hidden to output
				for (int i = 0; i < this.DIM_OF_HIDDEN + 1; i++) {
					for (int j = 0; j < this.DIM_OF_OUTPUT; j++) {
						weightChange2[i][j] = this.MOMENTUM_TERM * weightChange2[i][j] + this.LEARNING_RATE
								* this.getDeltaValue(j, this.HIDDEN_TO_OUTPUT, argValue) * this.hiddenNeurons[i];
						weightLayer2[i][j] += weightChange2[i][j];
					}
				}
		// update from input to hidden
		for (int i = 0; i < this.DIM_OF_INPUT; i++) {
			for (int j = 0; j < this.DIM_OF_HIDDEN; j++) {
				weightChange1[i][j] = this.MOMENTUM_TERM * weightChange1[i][j]
						+ this.LEARNING_RATE * this.getDeltaValue(j, this.INPUT_TO_HIDDEN, argValue) * inputVector[i];
				weightLayer1[i][j] += weightChange1[i][j];
			}
		}
		for (int j = 0; j < this.DIM_OF_HIDDEN; j++) {
			weightChange1[this.DIM_OF_INPUT][j] = this.MOMENTUM_TERM * weightChange1[this.DIM_OF_INPUT][j]
					+ this.LEARNING_RATE * this.getDeltaValue(j, this.INPUT_TO_HIDDEN, argValue)
							* NeuralNetInterface.BIAS;
			weightLayer1[this.DIM_OF_INPUT][j] += weightChange1[this.DIM_OF_INPUT][j];
		}
		
		return 0;
	}

	@Override
	public void save(File argFile) {
		// TODO Auto-generated method stub

	}

	@Override
	public void load(String argFileName) throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	@Override
	public double customSigmoid(double x) {
		return 0;
	}

	/**
	 * initialize the weights to [-0.5, 0.5]
	 */
	@Override
	public void initializeWeights() {
		for (int i = 0; i < weightLayer1.length; i++) {
			for (int j = 0; j < weightLayer1[0].length; j++) {
				weightLayer1[i][j] = Math.random() - 0.5;
			}
		}
		for (int i = 0; i < weightLayer2.length; i++) {
			for (int j = 0; j < weightLayer2[0].length; j++) {
				weightLayer2[i][j] = Math.random() - 0.5;
			}
		}
	}

	/**
	 * initialize all weights to zero;
	 */
	@Override
	public void zeroWeights() {
		for (int i = 0; i < weightLayer1.length; i++) {
			for (int j = 0; j < weightLayer1.length; j++) {
				weightLayer1[i][j] = 0;
			}
		}
		for (int i = 0; i < weightLayer2.length; i++) {
			for (int j = 0; j < weightLayer2.length; j++) {
				weightLayer1[i][j] = 0;
			}
		}
	}

	/**
	 * calculate the total error
	 * 
	 * @param trainingSet
	 *            all training data
	 * @return total error
	 * @throws InputFormatException
	 */
	public double getTotalError(List<TrainingData> trainingSet) throws InputFormatException {
		double totalError = 0;
		for (int i = 0; i < trainingSet.size(); i++) {
			double[] sampleResult = this.outputFor(trainingSet.get(i).getInputVector());
			double[] expectedOutput = trainingSet.get(i).getOutputVector();
			// System.out.println(sampleResult[0]+","+expectedOutput[0]);
			if (sampleResult.length != trainingSet.get(i).getOutputVector().length) {
				throw new InputFormatException("dimension of output vector mismatch!");
			}
			for (int j = 0; j < sampleResult.length; j++) {
				totalError += 0.5 * Math.pow(sampleResult[j] - expectedOutput[j], 2);
			}
		}
		return totalError;
	}

	/**
	 * calculate the delta in formula
	 * 
	 * @param index
	 *            namely 'j'
	 * @param whichLayer
	 *            indicate yj is output unit or hidden unit
	 * @param expectedOutput
	 *            Cj
	 * @return delta value
	 * @throws InputFormatException
	 */
	public double getDeltaValue(int index, int whichLayer, double[] expectedOutput) throws InputFormatException {
		double delta = 0;
		if (whichLayer == this.HIDDEN_TO_OUTPUT) {
			if (index >= this.DIM_OF_HIDDEN) {
				throw new InputFormatException("index out of bound!");
			}
			delta = this.diffActivFun(this.outputVector[index]) * (expectedOutput[index] - this.outputVector[index]);
		} else {
			if (index >= this.DIM_OF_HIDDEN) {
				throw new InputFormatException("index out of bound!");
			}
			for (int i = 0; i < this.DIM_OF_OUTPUT; i++) {
				delta += getDeltaValue(i, this.HIDDEN_TO_OUTPUT, expectedOutput) * weightLayer2[index][i];
			}
			delta *= this.diffActivFun(this.hiddenNeurons[index]);
		}
		return delta;
	}
	/**
	 * 
	 * @param trainingSet 	
	 * @param maxEpoch 		the criteria for non-converge, set 0 if you do not need it.
	 * @return -1 when no converge, total epoches when converge
	 * @throws InputFormatException
	 */
	public int trainingPlan(List<TrainingData> trainingSet, int maxEpoch) throws InputFormatException {
		if (maxEpoch == 0) {
			int epoch = 0;
			double totalError = this.getAcceptableError() + 1;
			while (totalError > this.getAcceptableError()) {
				for (int i = 0; i < trainingSet.size(); i++) {
					this.train(trainingSet.get(i).getInputVector(), trainingSet.get(i).getOutputVector());
				}
				epoch++;
				totalError = this.getTotalError(trainingSet);
				//System.out.println(totalError);
			}
			return epoch;
		}else {
			int epoch = 0;
			double totalError = this.getAcceptableError() + 1;
			while (totalError > this.getAcceptableError()&&epoch < maxEpoch) {
				for (int i = 0; i < trainingSet.size(); i++) {
					this.train(trainingSet.get(i).getInputVector(), trainingSet.get(i).getOutputVector());
				}
				epoch++;
				totalError = this.getTotalError(trainingSet);
				//System.out.println(totalError);
			}
			if(epoch >= maxEpoch && totalError > this.getAcceptableError()) {
				return -1;
			}else {
				return epoch;
			}
		}
	}

	public double getAcceptableError() {
		return acceptableError;
	}

	public void setAcceptableError(double acceptableError) {
		this.acceptableError = acceptableError;
	}

	/**
	 * customed activation funtion, you can replace with sigmoid in extended class;
	 * 
	 * @param x
	 *            input value
	 * @return
	 */
	abstract double activFunction(double x);

	abstract double diffActivFun(double fX);
}
