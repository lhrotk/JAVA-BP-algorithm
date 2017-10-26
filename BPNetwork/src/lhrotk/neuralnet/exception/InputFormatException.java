package lhrotk.neuralnet.exception;
/**
 * 
 * @author lhrotk
 *
 */
public class InputFormatException extends Exception{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public InputFormatException(){
        
    }
    public InputFormatException(String msg){
        super(msg);
    }
    public InputFormatException(String msg,Throwable cause){
        super(msg,cause);
    }
    public InputFormatException(Throwable cause){
        super(cause);
    }
    
    public static void main(String[] args) throws InputFormatException {
    	throw new InputFormatException("input dimension dismatch");
    }
}
