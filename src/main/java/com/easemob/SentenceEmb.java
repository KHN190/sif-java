package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SentenceEmb {

    private final static String vecFile = "glove.6B.50d.txt";
    private final static String wordFile = "idf.txt";
    private DataIO dataIO = new DataIO();

    private Map<String, Array2DRowRealMatrix> wordMap = dataIO.getWordVec(vecFile);
    private Map<String, Double> wordWeight = dataIO.getWordWeight(wordFile);

    private int wordVecLen = wordMap.get(")").getRowDimension();
    private Double defaultW = Collections.min(wordWeight.values());

    public SentenceEmb() throws IOException {}

    /* convert a list of words to a weighted vector, return [1 x wordVecLen] */
    public RealVector weightedAvg(List<String> text) {
        int sentLen = text.size();
        RealMatrix emb = new Array2DRowRealMatrix(sentLen, wordVecLen);
        RealVector w = new ArrayRealVector(sentLen);
        for (int i = 0; i < sentLen; i++) {
            String word = text.get(i);
            Double weight = wordWeight.get(word);
            if (weight == null) {
                weight = defaultW;
            }
            w.setEntry(i, weight);
            emb.setRowMatrix(i, wordMap.get(word).transpose());
        }
        RealVector res = emb.preMultiply(w).mapMultiply(1.0 / w.getDimension());
        return res;
    }

    /* calculate principle components */
    public RealMatrix getTruncatedSVD(RealMatrix m, int k) {
        SingularValueDecomposition svd = new SingularValueDecomposition(m);

        double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
        double[][] truncatedS = new double[k][k];
        double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];

        svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);
        svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);
        svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

        RealMatrix u = new Array2DRowRealMatrix(truncatedU);
        RealMatrix s = new Array2DRowRealMatrix(truncatedS);
        RealMatrix vt = new Array2DRowRealMatrix(truncatedVT);

        return u.multiply(s).multiply(vt);
    }

    /**
     * Convert a list of words to weighted vector and remove
     * the most common shared principle component(s).
     *
     * @param text tokenized sentence made of token strings
     * @param k remove how many principle components? (default 0)
     * @return embedded sentence using weights and word vector
     */
    public RealMatrix embedding(List<String> text, int k) {
        RealVector m = weightedAvg(text);
        RealMatrix res = new Array2DRowRealMatrix(1, m.getDimension());
        res.setRowVector(0, m);
        if (k > 0) {
            res = removePrincipleComponents(res, k);
        }
        return res;
    }

    public RealMatrix embedding(List<String> text) {
        return embedding(text, 0);
    }

    /**
     * Convert a list of sentences to weighted vectors and remove
     * the most common shared principle component(s).
     *
     * @param texts tokenized sentences made of token strings
     * @param k remove how many principle components? (default 1)
     * @return embedded sentences using weights and word vector
     */
    public RealMatrix matrixEmbedding(List<List<String>> texts, int k) {
        RealMatrix res = new Array2DRowRealMatrix(texts.size(), wordVecLen);
        for (int i = 0; i < texts.size(); ++i) {
            List<String> text = texts.get(i);
            res.setRowMatrix(i, embedding(text, 0));
        }
        if (k > 0) {
            res = removePrincipleComponents(res, k);
        }
        return res;
    }

    public RealMatrix matrixEmbedding(List<List<String>> texts) {
        return matrixEmbedding(texts, 1);
    }

    /* remove principle components */
    private RealMatrix removePrincipleComponents(RealMatrix m, int k) {
        RealMatrix pc = getTruncatedSVD(m, k);
        return m.subtract(m.multiply(pc.transpose()).multiply(pc));
    }
}
