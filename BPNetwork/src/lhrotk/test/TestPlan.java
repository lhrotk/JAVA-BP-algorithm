package lhrotk.test;
/**
 * 
 * @author lhrotk
 *
 */
public class TestPlan {
	private int max;
	private int min;
	private int count;
	private int nonConverge;
	private int total;
	public int getMax() {
		return max;
	}
	public void setMax(int max) {
		this.max = max;
	}
	public int getMin() {
		return min;
	}
	public void setMin(int min) {
		this.min = min;
	}
	public int getCount() {
		return count;
	}
	public void setCount(int count) {
		this.count = count;
	}
	public double getAverage() {
		return ((double)total)/count;
	}
	
	public void addNewData(int data) {
		if(count == 0) {
			max = min = total = data;
		}
		if(data>max) {
			max = data;
		}
		if(data<min) {
			min = data;
		}
		count++;
		total	+= data;
	}
	
	public void addNewNonConverge() {
		this.nonConverge++;
	}
	
	public double getNonConvergeRate() {
		return((double)(this.nonConverge))/(this.nonConverge + this.count);
	}
}
