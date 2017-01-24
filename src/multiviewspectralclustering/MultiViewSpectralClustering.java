/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multiviewspectralclustering;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author pavan
 */
public class MultiViewSpectralClustering {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        int num_views=3;
        int numClust=2;
        int numIter=5;
        
        double[] sigma = new double[3];
        double sigma1 = sigma[0] = 2.565656085779173;
        double sigma2 = sigma[1] = 0.888739619532634;
        double sigma3 = sigma[2] = 2.213400238591733;
        
        String x1Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X1.txt";
        String x2Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X2.txt";
        String x3Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X3.txt";
        
        double[][] X1 =  readMatrix(x1Path);
        System.out.println("done 1");
        double[][] X2 =  readMatrix(x2Path);
        System.out.println("done 2");
        double[][] X3 =  readMatrix(x3Path);
        
        
        System.out.println("Single best view ");
        for(int i=0;i<num_views;i++)
        {
            System.out.println("With View "+(i+1));
            
        }
        
        
    }
    
    
    public static void constructKernel(double[][] m,double sigma)
    {
        
    }
    
    public static double[][] readMatrix(String filePath) throws FileNotFoundException, IOException, Exception
    {
        FileReader fr =  new FileReader(filePath);
        BufferedReader br =  new BufferedReader(fr);
        double[][] inputArray = new double[1000][2];
        for(int i=0;i<1000;i++){
            String str =  br.readLine();
            if(str ==  null)
                throw new Exception("Invalid line in the file");
            String[] columnValues = str.split("	");
            for(int j=0;j<2;j++){
                inputArray[i][j]= Double.parseDouble(columnValues[j]); 
            }
            
       }
       return inputArray;
       
    }
    
    public static double optSigma(double[][] X)
    {
        int N = 1000;
        double dist = EuDist2(X,X);
        return 0.0;
    }
    
    public static double EuDist2(double[][] m1, double[][] m2)
    {
        //implementation 
        return 0.0;
    }
}
