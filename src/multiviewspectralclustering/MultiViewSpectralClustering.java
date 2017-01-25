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
import static java.lang.Math.sqrt;
import java.lang.reflect.Array;
import java.util.Arrays;

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
//        double sigma1 = sigma[0] = 2.565656085779173;
//        double sigma2 = sigma[1] = 0.888739619532634;
//        double sigma3 = sigma[2] = 2.213400238591733;
//        
        String x1Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X1.txt";
        String x2Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X2.txt";
        String x3Path = "/Users/pavan/NetBeansProjects/MultiViewSpectralClustering/src/multiviewspectralclustering/X3.txt";
        
        double[][] X1 =  readMatrix(x1Path);
        System.out.println("done 1");
        double[][] X2 =  readMatrix(x2Path);
        System.out.println("done 2");
        double[][] X3 =  readMatrix(x3Path);
               
//        double[ ] num[ ] = {{1,2}, {2,1}, {3,3}};
//        double[][] x =EuDist2(num,num);
        double sigma1 = optSigma(X1);
        double sigma2 = optSigma(X2);
        double sigma3 = optSigma(X3);
        
        
//        System.out.println("Single best view ");
//        for(int i=0;i<num_views;i++)
//        {
//            System.out.println("With View "+(i+1));
//            
//        }
//        
        
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
        int N = X.length;
        double[][] dist = EuDist2(X,X);
        double[] newdist = new double[N*N];
        int i=0;
        double median = 0;
        for(int j=0;j<dist[0].length;j++)
            for(int k=0;k<dist.length;k++)
            {
                newdist[i] = dist[j][k];
                i++;
            }
        
        Arrays.sort(newdist);
        if(N*N%2 == 0)
        {
            int val = (int)N*N/2;
            median =  (newdist[val-1] + newdist[val])/2;
        }
        else
        {
            int val = (int)N*N/2;
            median = newdist[val+1];
        }
        System.out.println("median "+median);
        return 0.0;
    }
    
    public static double[][] EuDist2(double[][] m1, double[][] m2)
    {
        int bSqrt =1;
        
        int nSmp_a = m1.length,nSmp_b =m2.length;
        int nFea = m1[0].length;
        
        double[][] aa = sumMatrix(m1);
        double[][] bb = sumMatrix(m2);
        
        
        double[][] m2T = new double[nFea][nSmp_b];
        
        for(int i=0;i<nFea;i++)
            for(int j=0;j<nSmp_b;j++)
                m2T[i][j] =  m2[j][i];
        
        double[][] ab = multiMatrix(m1, m2T);

//        printMatrix(aa);
//        printMatrix(bb);
//        printMatrix(ab);
//        
        double[][] bbT = new double[bb[0].length][bb.length];
        
        for(int i=0;i<bb[0].length;i++)
            for(int j=0;j<bb.length;j++)
                bbT[i][j] =  bb[j][i];
        
        double[][] aarepmat = new double[nSmp_a][nSmp_b];
        double[][] bbrepmat = new double[nSmp_a][bbT[0].length];
        
        for(int i=0;i<aarepmat.length;i++)
            for(int j=0;j<aarepmat[0].length;j++)
            {
                aarepmat[i][j]=aa[i][0];
            }
        
//        printMatrix(aarepmat);
        
        for(int i=0;i<bbrepmat.length;i++)
            for(int j=0;j<bbrepmat[0].length;j++)
            {
                bbrepmat[i][j]=bbT[0][j];
            }
        
      
        double[][] resMat = new double[nSmp_a][nSmp_b];
        for(int i=0;i<nSmp_a;i++)
            for(int j=0;j<nSmp_b;j++) {
                resMat[i][j] = Math.abs(sqrt(aarepmat[i][j]+bbrepmat[i][j]-2*ab[i][j]));
            }
        
        return resMat;
    }
    
    public static void printMatrix(double[][] m){
         for(int i=0;i<m.length;i++) {
            for(int j=0;j<m[0].length;j++)
                System.out.print(" "+m[i][j]);
            System.out.println("");
        }
    }
    
    public static double[][] multiMatrix(double[][] m1,double[][] m2)
    {
        int nSmp_a = m1.length,nSmp_b =m2.length;
        int nFea_a = m1[0].length;
        int nFea_b = m2[0].length;
//        System.out.println(" "+nSmp_a+" "+nFea_a+" "+nSmp_b+" "+nFea_b);
        double[][] ab = new double[nSmp_a][nFea_b];
        for (int i = 0; i < nSmp_a; i++) {
            for (int j = 0; j < nFea_b; j++) {
                ab[i][j] = 0.00000;
            }
        }

        for (int i = 0; i < nSmp_a; i++) { // aRow
            for (int j = 0; j < nFea_b; j++) { // bColumn
                for (int k = 0; k < nFea_a; k++) { // aColumn
                    ab[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return ab;
    }
    
    public static double[][] sumMatrix(double[][] m)
    {
        int nSmp = m.length;
        double[][] sumMat = new double[nSmp][1];
        for(int i=0;i<nSmp;i++)
        {
            double sum =0;
            for(int j=0;j<2;j++)
            {
                sum += m[i][j]*m[i][j];
            }
            sumMat[i][0]=sum;
        }
        return sumMat;
    }
}
