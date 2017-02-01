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


import Jama.*;
import Jama.EigenvalueDecomposition;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
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
        int numiter=10;
        double projev=1.5;
//        double[] sigma = new double[3];
//        double sigma1 = sigma[0] = 2.565656085779173;
//        double sigma2 = sigma[1] = 0.888739619532634;
//        double sigma3 = sigma[2] = 2.213400238591733;
        
        String x1Path = "/Users/pavan/Documents/Knoesis/Thesis/"
                + "MultiViewSpectralClustering/src/multiviewspectralclustering/X1.txt";
        String x2Path = "/Users/pavan/Documents/Knoesis/Thesis/"
                + "MultiViewSpectralClustering/src/multiviewspectralclustering/X2.txt";
        String x3Path = "/Users/pavan/Documents/Knoesis/Thesis/"
                + "MultiViewSpectralClustering/src/multiviewspectralclustering/X3.txt";
        String truthPath = "/Users/pavan/Documents/Knoesis/Thesis"
                + "/MultiViewPIC/src/InputFiles/truth.txt";
        double[][] X1 =  readMatrix(x1Path);
        
        double[][] X2 =  readMatrix(x2Path);
        
        double[][] X3 =  readMatrix(x3Path);
        
        FileReader fr =  new FileReader(truthPath);
        BufferedReader br =  new BufferedReader(fr);
        
        int[] truth = new int[1000];
        String str =  br.readLine();
        for(int i=0;i<1000;i++){
            str =  br.readLine();
            if(str ==  null)
                throw new Exception("Invalid line in the file");
            truth[i] = Integer.parseInt(str);
       }
        
//        double[][] truthTemp =  readMatrix(truthPath);
        
//        for(int i=0;i<truthTemp.length;i++)
//            truth[i] =  (int)truthTemp[i][0];
        
//        double[][] X2 = { {1,2},{2,4},{1,4},{6,5},{4,2},{4,3},{2,5},{2,2},{1,1},{3,1}};
//        double[][] X1 = { {4,3},{2,5},{2,2},{1,1},{3,1},{6,5},{3,4},{2,3},{2,1},{3,2}};
//        double[][] X3 = { {6,5},{3,4},{2,3},{2,1},{3,2},{1,2},{2,4},{1,4},{6,5},{4,2}};
//        int[] truth = {2,1,2,2,1,2,1,1,2,2};
//        double[ ] num[ ] = {{1,2,3}, {2,1,1}, {3,3,2}};
//        double[][] x =EuDist2(num,num);
        double sigma1 = optSigma(X1);
        double sigma2 = optSigma(X2);
        double sigma3 = optSigma(X3);
        
        List<double[][]> data = new ArrayList<double[][]>();
        
        
        data.add(X1);
        data.add(X2);
        data.add(X3);
       
        double[] sigma = {sigma1,sigma2,sigma3};
       
        
        spectral_cotraining(data,num_views,numClust,sigma,truth,projev,numiter);
        
        /*
        List<int[]> C = new ArrayList<int[]>();
        List<double[][]> V = new ArrayList<double[][]>();
        List<double[][]> K = new ArrayList<double[][]>();
        
        for(int i=0;i<num_views;i++)
        {
            System.out.println("With View "+(i+1));
            
            K.add(constructKernel(data.get(i), data.get(i), sigma[i]));
            ArrayList basRes = baseline_spectral_onkernel(K.get(i),numClust,truth,projev);
            V.add((double[][])basRes.get(1));
            C.add((int[])basRes.get(0));
        }
        */
       

//        for (int i=0; i<num.length; ++i) 
//        {
//            for (int j=0; j<num[0].length; ++j) 
//            {
//                double c0 = Math.abs(num[i][j]);
//                c0 = sqrt(c0);
//                System.out.print(" "+c0);
//                
//            }
//            System.out.println("");
//        }

    }
    
    public static void spectral_cotraining( List<double[][]> data,int num_views,
            int numClust,double[] sigma,int[] truth,double projev,int numiter)
    {
        int[] truthVector = truth;
        
        List<double[][]> K = new ArrayList<double[][]>();
        List<int[]> C = new ArrayList<int[]>();
        List<double[][]> V = new ArrayList<double[][]>();
        
        int N = data.get(0).length;
        for(int i=0;i<num_views;i++)
        {
            System.out.println("Computing kernel for view "+(i+1));
            K.add(constructKernel(data.get(i), data.get(i), sigma[i]));
            ArrayList basRes = baseline_spectral_onkernel(K.get(i),numClust,truth,projev);
//            System.out.println(" printing V return values");
//            printMatrix((double[][])basRes.get(1));
                System.out.println(" nmi "+(double)basRes.get(2));
            V.add((double[][])basRes.get(1));
            C.add((int[])basRes.get(0));
        }
        
        
       
        List<double[][]> X = new ArrayList<double[][]>();
        List<double[][]> Y = new ArrayList<double[][]>();
        List<double[][]> Y_norm = new ArrayList<double[][]>();
        
        X = V;
        Y = K;
        Y = Y_norm;
        
        double[][] Sall = new double[K.get(0).length][K.get(0).length];
        double[][] SallTemp = new double[K.get(0).length][K.get(0).length];
        double[][] mRes = new double[K.get(0).length][K.get(0).length];
        
        
        System.out.println("Starting Co-training approach");
        for(int i=0;i<1;i++)
        {
            
            System.out.println("\nIteration ..."+ (i+1));
            
            for(int ll=0;ll<N;ll++)
                for(int lm=0;lm<N;lm++)
                    Sall[ll][lm]=0;
//            System.out.println("print Sall");
//            printMatrix(Sall);
            
            DecimalFormat df = new DecimalFormat("#.########");
            df.setRoundingMode(RoundingMode.CEILING);
            
                    
            Matrix Sallmat = new Matrix(Sall);
            for(int j=0;j<num_views;j++)
            {
//                System.out.println("Print Xtmp");
                
//                printMatrix(Xtmp);
//                for(int ll=0;ll<Xtmp.length;ll++)
//                    for(int lm=0;lm<Xtmp[0].length;lm++)
//                        Xtmp[ll][lm] = Double.valueOf(df.format(Xtmp[ll][lm]));
                double[][] Xtmp = X.get(j);
                Matrix Xmat = new Matrix(Xtmp);
                Matrix Tmat = Xmat.times(Xmat.transpose());
//                printMatrix(Tmat.getArray());
                Sallmat =  Sallmat.plus(Tmat);
                Sall = Sallmat.getArray();
                
              /*
                System.out.println("print X "+j);
//                printMatrix((double[][])X.get(j));
                double[][] tempTrans = transposeMatrix((double[][])X.get(j));
//                System.out.println(" temp tans "+tempTrans.length+" column "+tempTrans[0].length);
//                System.out.println("print X trans "+j);
//                printMatrix(tempTrans);
                mRes = multiMatrix((double[][])X.get(j),transposeMatrix((double[][])X.get(j)));
//                System.out.println("print mres"+j);
//                printMatrix(mRes);
                Sall = sumMatrix(Sall,mRes);
                System.out.println("mormal print Sall ");
                printMatrix(Sall);
               */ 

            }
                
                
            for(int j=0;j<num_views;j++)
            {
                double[][] Xtmp = X.get(j);
                Matrix Xmat = new Matrix(Xtmp);
                
                double[][] Ktmp = K.get(j);
                Matrix Kmat = new Matrix(Ktmp);
                
                Matrix Tmat = Xmat.times(Xmat.transpose());
//                System.out.println(" printing Xmat");
//                printMatrix(Kmat.getArray());
                
                Matrix Submat = Sallmat.minus(Tmat);
                Matrix Ytemp = Kmat.times(Submat);
//                System.out.println(" Input ");
//                printMatrix(Ytemp.getArray());
                
                Ytemp = Ytemp.plus(Ytemp.transpose());
//                Y.add(Ytemp.getArray());
                double[][] YnTmp = Ytemp.getArray();
                for(int ll=0;ll<YnTmp.length;ll++)
                    for(int lm=0;lm<YnTmp.length;lm++)
                        YnTmp[ll][lm] = YnTmp[ll][lm]/2;
                
                Y.add(YnTmp);
                
//                System.out.println(" Y values are  ");
//                printMatrix(YnTmp);
                Y_norm.add(YnTmp);
//                System.out.println("View ..."+(j+1));
                ArrayList basRes  =  baseline_spectral_onkernel(YnTmp,numClust,truthVector,projev);
                System.out.println(" nmi "+(double)basRes.get(2));
//                printMatrix((double[][])basRes.get(1));
                X.set(j, (double[][])basRes.get(1));
                C.set(j, (int[])basRes.get(0));
               
//                mRes = multiMatrix((double[][])X.get(j),transposeMatrix((double[][])X.get(j)));
//                Sall = subMatrix(Sall, mRes);
//                double[][] newCalRes =multiMatrix((double[][])K.get(j), mRes);
//                Y.add(newCalRes);
//                SallTemp = sumMatrix((double[][])Y.get(j), transposeMatrix((double[][])Y.get(j)));
//                for(int ia=0;ia<mRes.length;ia++)
//                    for(int ib=0;ib<mRes[0].length;ib++)
//                    {
//                        SallTemp[ia][ib] = SallTemp[ia][ib]/2;
//                    }
//                System.out.println("Printing SallTemp");
//                printMatrix(SallTemp);
//                Y.set(j,SallTemp);
//                Y_norm.add(Y.get(j) );
//                ArrayList basRes  =  baseline_spectral_onkernel(Y_norm.get(j),numClust,truthVector,projev);
//                X.set(j, (double[][])basRes.get(1));
//                C.set(j, (int[])basRes.get(0));

                 

                


            }
            
            
        }
        
        
    }
    
     public static double[][] transposeMatrix(double[][] m)
    {
        double[][] res = new double[m[0].length][m.length];
        for(int i=0;i<m.length;i++)
            for(int j=0;j<m[0].length;j++)
                res[j][i]=m[i][j];
        return res;
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
    
      public static double[][] sumMatrix(double[][] m1,double[][] m2)
    {
        int row  = m1.length;
        int column = m1[0].length;
        
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<column;j++)
            {
                m1[i][j] = m1[i][j]+m2[i][j];
            }
        }
        return m1;
    }
    
    public static double[][] subMatrix(double[][] m1,double[][] m2)
    {
        int row  = m1.length;
        int column = m1[0].length;
        
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<column;j++)
            {
                m1[i][j] = m1[i][j]-m2[i][j];
            }
        }
        return m1;
    }
    
    
    public static ArrayList baseline_spectral_onkernel(double[][] K,int numClust,int[] truth,double projev)
    {
//        printMatrix(K);
//        double[] nmi = new double[K.length];
        System.out.println("printing K");
        printMatrix(K);
        int numEv = (int) Math.ceil(numClust * projev);
        double[][] V = new double[K.length][numEv];
        int N =K.length;
        double[][] diag = new double[K.length][K.length];
        
        for(int i=0;i<K.length;i++)
        {
            double sum=0;
            for(int j=0;j<K[0].length;j++)
            {
                sum +=K[i][j];
                diag[i][j] = 0; 
            }
            diag[i][i] = sum;
        }
        
//        printMatrix(diag);
                
//        Matrix A = new Matrix(num);
//        Matrix B = new Matrix(diag);
//        Matrix C = A.times(B);
//        num = C.getArray();
//        printMatrix(num);
        long startTime = System.currentTimeMillis();
        Matrix A = new Matrix(diag);
        Matrix inv_A = A.inverse();
        double[][] inv = inv_A.getArray();
        for(int i=0;i<inv.length;i++)
            for(int j=0;j<inv[0].length;j++)
                inv[i][j] = Math.sqrt( Math.abs(inv[i][j]) );
//        printMatrix(inv);
        Matrix matK =  new Matrix(K);
        Matrix inv_sqrt_D =  new Matrix(inv);
        
        Matrix temp = inv_sqrt_D.times(matK);
        Matrix matL = temp.times(inv_sqrt_D);
        matL =  matL.plus(matL.transpose());
        
        double[][] L = matL.getArray();
        for(int i=0;i<L.length;i++)
            for(int j=0;j<L[0].length;j++)
                L[i][j] = L[i][j]/2;
//        printMatrix(L);
        matL = new Matrix(L);
        EigenvalueDecomposition matE=matL.eig();
        Matrix matV = matE.getV();
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
//        System.out.println("SC time "+elapsedTime);
        double[][] VmatTemp = matV.getArray();
//        printMatrix(V);
        double[][] U = new double[VmatTemp.length][numClust];
        for(int i=0;i<VmatTemp.length;i++)
            for(int j=VmatTemp[0].length-1;j>VmatTemp[0].length-numClust-1;j--)
                U[i][VmatTemp[0].length-j-1] = VmatTemp[i][j];
        
        double[][] tempV = new double[VmatTemp.length][numEv];
        for(int i=0;i<VmatTemp.length;i++)
            for(int j=VmatTemp[0].length-1;j>VmatTemp[0].length-1-numEv;j--)
                tempV[i][VmatTemp[0].length-j-1] = VmatTemp[i][j];
        V =  tempV;
//        System.out.println(" V is");
//        printMatrix(tempV);
//        System.out.println(" V is");
//        printMatrix(V);
        double[][] norm_mat = new double[VmatTemp.length][numClust];
        
        double sqVal=0;
        for(int l=0;l<K.length;l++)
        {
            sqVal=0;
            for(int m=0;m<2;m++)
            {
                sqVal = sqVal+U[l][m]*U[l][m];
            }
            sqVal = Math.sqrt(sqVal);
            for(int n=0;n<2;n++)
            {
                if(sqVal == 0)
                    sqVal =1;
                norm_mat[l][n] = sqVal;
            }
        }
        
        DecimalFormat df = new DecimalFormat("#.########");
        df.setRoundingMode(RoundingMode.CEILING);
      
        for(int l=0;l<K.length;l++)
        {
            for(int m=0;m<2;m++)
            {
                U[l][m] = U[l][m]/norm_mat[l][m];
                U[l][m] = Double.valueOf(df.format(U[l][m]));
            }
        }
//        System.out.println("Normalized matrix ");
//        printMatrix(U);
        
        int[] C = new int[K.length];
        
        System.out.println("running k-means...");
        double maxnmi =0;
        for(int b=0;b<10;b++)
        {
//            System.out.println("Kmean at iteration "+b);
            Kmean KM = new Kmean(U,null);
            int[] H = KM.clustering(2, 10, null); // 2 clusters, maximum 10 iterations
//            KM.printResults();
            
//            compute_f(truth,H);
//            
            Integer[] Tnew = Arrays.stream(truth).boxed().toArray( Integer[]::new);
            Integer[] Hnew = Arrays.stream(H).boxed().toArray( Integer[]::new);
//        
            ArrayList nmiVAL = compute_nmi(Tnew,Hnew);
            if(maxnmi < (double)nmiVAL.get(1));
                maxnmi=(double)nmiVAL.get(1);
            C=H;
            
        }
        
//        System.out.println(" Print v in base line");
        
        ArrayList retVal =  new ArrayList();
        retVal.add(C);
        retVal.add(V);
        retVal.add(maxnmi);
        return retVal;
    }
    
      public static ArrayList compute_nmi(Integer[] T,Integer[] H)
    {
        int N = T.length;
        double avgent =0;
        Set<Integer> uniqClasses = new TreeSet<Integer>();
        uniqClasses.addAll(Arrays.asList(T));
        Integer[] classes = uniqClasses.toArray(new Integer[0]);
        
        Set<Integer> uniqClusters = new TreeSet<Integer>();
        uniqClusters.addAll(Arrays.asList(H));
        Integer[] clusters = uniqClusters.toArray(new Integer[0]);
        
//        System.out.println("uniqKeys: " + uniqClasses);
//        System.out.println(" "+uniqClasses.size());
//        
//        System.out.println("uniqKeys: " + uniqClusters);
//        System.out.println(" "+uniqClusters.size());
       
        int num_class = uniqClasses.size();
        int num_clust = uniqClusters.size();
        
        Integer[] index_class = new Integer[T.length];
        Integer[] index_clust = new Integer[H.length];
        
        double[] D =  new double[num_class];
        double[] B =  new double[num_clust];
        
        for(int j=0;j<num_class;j++)
        {
            int count =0;
            for(int i=0;i<T.length;i++)
            {
                if(T[i] == classes[j])
                {
                    index_class[i]=1;
                    count++;
                }
                else
                    index_class[i]=0;
            }
            D[j]=count;
        }
        
        double mi=0;
        double[][] A = new double[num_clust][num_class];
        double[][] miarr = new double[num_clust][num_class];
        
        for(int i=0;i<num_clust;i++)
            for(int j=0;j<num_class;j++)
                A[i][j]=0;
        
        int avgen=0;
        
        for(int i=0;i<num_clust;i++)
        {
            int count =0;
            for(int j=0;j<H.length;j++)
            {
                if(H[j] == clusters[i])
                {
                    index_clust[j]=1;
                    count++;
                }
                else
                    index_clust[j]=0;
            }
            B[i]=count;
            
            for(int j=0;j<num_class;j++)
            {
                for(int k=0;k<T.length;k++)
                {
                    if(T[k] == classes[j])
                    {
                        index_class[k]=1;
                    }
                    else
                        index_class[k]=0;
                }
                int ccount=0;
                for(int l=0;l<T.length;l++)
                {
                    index_class[l] = index_class[l]*index_clust[l];
                    if(index_class[l] == 1)
                        ccount++;
                }
                
                A[i][j] = ccount;
                if(A[i][j] != 0)
                {
                    miarr[i][j] = (A[i][j]/N) * ((Math.log( (N*A[i][j])/
                            (B[i]*D[j]) ))/(Math.log(2)));
                    avgent = avgent - ( (B[i]/N) * (A[i][j]/B[i]) * 
                            (( Math.log(A[i][j]/B[i]))/Math.log(2)) ); 
                }
                else
                {
                    miarr[i][j] = 0;
                }
                mi = mi + miarr[i][j];
            }
            
            
        }
//        System.out.println("mi "+mi);
        
        double class_ent=0;
        double clust_ent=0;
        
        for(int i=0;i<num_class;i++)
            class_ent = class_ent + (D[i]/N)*((Math.log(N/D[i]))/(Math.log(2)));
        
        for(int i=0;i<num_clust;i++)
            clust_ent = clust_ent + (B[i]/N)*((Math.log(N/B[i]))/(Math.log(2)));
        
        double nmi = 2*mi/(clust_ent+class_ent);
        
        
        
//        System.out.println("nmi "+nmi);
        ArrayList rVal =  new ArrayList();
        rVal.add(A);
        rVal.add(nmi);
        rVal.add(avgent);
        return rVal;
    }
    
    
    public static void compute_f(int[] truth,int[] H)
    {
        int N = truth.length;
        int numT = 0;
        int numH = 0;
        int numI = 0;
//        System.out.println("N is "+N);
        for(int n=0;n<N;n++)
        {
            double[] TnVector =  new double[N-n-1];
            int sumTn=0,sumHn=0;
            int count=0;
//            System.out.println("\n TnVector");
            for(int i=n+1;i<N;i++) {
                if(truth[n] == truth[i])
                {    
                    TnVector[count]=1;
                    sumTn++;
                }
                else
                    TnVector[count]=0;
//                System.out.print(" "+TnVector[count]);
                count++;
            }
//            System.out.println("\n HnVector");
            double[] HnVector =  new double[N-n-1];
            count =0;
            for(int i=n+1;i<N;i++) {
                if(H[n] == H[i])
                {
                    HnVector[count]=1;
                    sumHn++;
                }
                else
                    HnVector[count]=0;
//                System.out.print(" "+HnVector[count]);
                count++;
            }
            
            numT = numT + sumTn;
            numH = numH + sumHn;
//            System.out.println("\n sumTn "+sumTn+" sumHn "
//                    +sumHn+" numT "+numT+" numH "+numH);
            
            int sumIn=0;
//            System.out.println(" mulVector ");
            for(int i=0;i<N-n-1;i++){
                TnVector[i] =  TnVector[i]*HnVector[i];
                if(TnVector[i] == 1)
                    sumIn++;
//                System.out.print(" "+TnVector[i]);
            }
            numI = numI + sumIn;
//            System.out.println("\n sumIn "+sumIn+" numI "+numI);
        }
        double p =1;
        double r =1;
        double f =1;
        if (numH > 0)
           p = (double)numI/numH;
        if (numT > 0)
           r = (double)numI/numT;
//        System.out.println(" P "+p+" r "+r);
        if(p+r == 0)
            f =0;
        else
            f = (2*p*r)/(p+r);
        System.out.println("f is "+f);
    }
    
    
    public static double[][] constructKernel(double[][] m1,double[][] m2,double sigma)
    {
//        System.out.println(" "+sigma);
        double[][] D = EuDist2(m1,m2,0);
        double[][] K = new double[D.length][D[0].length];
//        System.out.println("D ");
        for(int i=0;i<D.length;i++)
            for(int j=0;j<D[0].length;j++)
                K[i][j] = Math.exp((-(D[i][j]))/(2*(Math.pow(sigma,2))));
//        printMatrix(K);
        return K;
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
//        printMatrix(X);
        double[][] dist = EuDist2(X,X,1);
//        System.out.println(" dist is");
//        printMatrix(dist);
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
//        System.out.println("The sorted int array is:");
//        for (double number : newdist) {
//            System.out.print(" " + number);
//            System.out.println("");
//        }
        if(N*N%2 == 0)
        {
            int val = (int)N*N/2;
            median =  (newdist[val-1] + newdist[val])/2;
        }
        else
        {
            int val = (int)N*N/2;
            median = newdist[val];
        }
//        System.out.println("median "+median);
        return median;
    }
    
    public static double[][] EuDist2(double[][] m1, double[][] m2,int val)
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
////        
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
        
        if(val == 1 )
        {
            for(int i=0;i<nSmp_a;i++)
                for(int j=0;j<nSmp_b;j++) {
                    resMat[i][j] = Math.abs(sqrt(aarepmat[i][j]+bbrepmat[i][j]-2*ab[i][j]));
                }
        }
        else
        {
           for(int i=0;i<nSmp_a;i++)
                for(int j=0;j<nSmp_b;j++) {
                    resMat[i][j] = Math.abs(aarepmat[i][j]+bbrepmat[i][j]-2*ab[i][j]);
                } 
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
