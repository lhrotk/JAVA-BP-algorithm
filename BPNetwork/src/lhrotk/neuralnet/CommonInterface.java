package lhrotk.neuralnet;
import java.io.File;
import java.io.IOException;

import lhrotk.neuralnet.exception.InputFormatException;
/**
* This interface is common to both the Neural Net and LUT interfaces.
* The idea is that you should be able to easily switch the LUT
* for the Neural Net since the interfaces are identical.
* @date 20 June 2012
* @author sarbjit
*
*/
public interface CommonInterface {
/**
* make prediction
* @param inputVector The input vector. An array of doubles.
* @return The value returned by th LUT or NN for this input vector
 * @throws InputFormatException 
*/
	public double[] outputFor(double [] inputVector) throws InputFormatException;
/**
* This method will tell the NN or the LUT the output
* value that should be mapped to the given input vector. I.e.
* the desired correct output value for an input.
* @param inputVector The input vector
* @param argValue The new value to learn
* @return The error in the output for that input vector
 * @throws InputFormatException 
*/
	public double train(double [] inputVector, double[] argValue) throws InputFormatException;
/**
* A method to write either a LUT or weights of an neural net to a file.
* @param argFile of type File.
*/
	public void save(File argFile);
/**
* Loads the LUT or neural net weights from file. The load must of course
* have knowledge of how the data was written out by the save method.
* You should raise an error in the case that an attempt is being
* made to load data into an LUT or neural net whose structure does not match
* the data in the file. (e.g. wrong number of hidden neurons).
* @param argFileName the name or argeFile
* @throws IOException
*/
	public void load(String argFileName) throws IOException;
}