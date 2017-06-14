package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TextualSim {

    private SentenceEmb sentenceEmb = new SentenceEmb();

    public TextualSim() throws IOException {
    }

    /* evaluate textual similarity between 2 sentences */
    public double score(List<String> t1, List<String> t2, int k) {
        RealMatrix e1 = sentenceEmb.embedding(t1, k);
        RealMatrix e2 = sentenceEmb.embedding(t2, k);
        // calculate cosine angle
        RealMatrix inn = inner(e1, e2);
        RealMatrix e1Norm = sqrt(inner(e1, e1));
        RealMatrix e2Norm = sqrt(inner(e2, e2));
        return div(div(inn, e1Norm), e2Norm).getEntry(0, 0);
    }

    public double score(List<String> t1, List<String> t2) {
        return score(t1, t2, 0);
    }

    /* evaluate textual similarity between pairs of sentences */
    public List<Double> scores(List<List<String>> t1, List<List<String>> t2, int k) {
        int s1 = t1.size();
        int s2 = t2.size();
        if (s1 != s2) {
            throw new RuntimeException("sentences must be in pairs, size: ("+ s1 +", " + s2 + ")");
        }
        // removed principle components
        for (int i = 0; i< s1; ++i) {
            RealMatrix e1 = sentenceEmb.matrixEmbedding(t1, k);
            RealMatrix e2 = sentenceEmb.matrixEmbedding(t2, k);
        }
        // calculate cosine angles
        List<Double> res = new ArrayList<>();
        for (int i = 0; i < s1; i++) {
            double s = score(t1.get(i), t2.get(i));
            res.add(s);
        }
        return res;
    }

    public List<Double> scores(List<List<String>> t1, List<List<String>> t2) {
        return scores(t1, t2, 1);
    }

    /* [m x s], [m x s] -> [1, m] */
    private RealMatrix inner(RealMatrix m1, RealMatrix m2) {
        if (m1.getColumnDimension() != m2.getColumnDimension() || m1.getRowDimension() != m2.getRowDimension()) {
            throw new RuntimeException("m1, m2 must be the same shape");
        }
        int m = m1.getRowDimension();
        int n = m2.getColumnDimension();
        // multiply m1, m2 element by element
        RealMatrix tmp = new Array2DRowRealMatrix(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double x = m1.getEntry(i, j) * m2.getEntry(i, j);
                tmp.setEntry(i, j, x);
            }
        }
        // sum over along axis 1
        RealMatrix res = new Array2DRowRealMatrix(1, m);
        for (int i = 0; i < m; ++i) {
            double sum = Arrays.stream(tmp.getRowVector(i).toArray()).sum();
            res.setEntry(0, i, sum);
        }
        return res;
    }

    /* sqrt m by element */
    private RealMatrix sqrt(RealMatrix m) {
        for (int i = 0; i < m.getRowDimension(); ++i) {
            for (int j = 0; j < m.getColumnDimension(); ++j) {
                m.setEntry(i, j, Math.sqrt(m.getEntry(i, j)));
            }
        }
        return m;
    }

    /* divide m1 by element in m2 accordingly, e.g. [3,6,9] / [3,3,3] -> [1,2,3] */
    private RealMatrix div(RealMatrix m1, RealMatrix m2) {
        int r1 = m1.getRowDimension();
        int r2 = m2.getRowDimension();
        int c1 = m1.getColumnDimension();
        int c2 = m2.getColumnDimension();
        if (r1 != r2 || c1 != c2) {
            throw new RuntimeException("m1, m2 must be the same shape (["+ r1 +" x "+ c1 +"], ["+ r2 +" x "+ c2 +"])");
        }
        RealMatrix res = new Array2DRowRealMatrix(r1, c1);
        for (int i = 0 ; i < r1; i++) {
            for (int j = 0; j < c1; j++) {
                double x = m1.getEntry(i, j) / m2.getEntry(i, j);
                res.setEntry(i, j, x);
            }
        }
        return res;
    }
}