package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TextualSimTest {

    private TextualSim sim = new TextualSim();

    public TextualSimTest() throws IOException {
    }

    @Test
    public void testScore() {
        List<String> s1 = new ArrayList<>();
        List<String> s2 = new ArrayList<>();
        List<String> s3 = new ArrayList<>();

        s1.add("polar");
        s1.add("bear");
        s1.add("is");
        s2.add("white");

        s1.add("polar");
        s2.add("bear");
        s2.add("is");
        s2.add("gray");

        s3.add("how");
        s3.add("about");
        s3.add("the");
        s3.add("weather");

        System.out.println(sim.score(s1, s1));
        System.out.println(sim.score(s1, s2));
        System.out.println(sim.score(s1, s3));
        System.out.println(sim.score(s2, s3));

        List<List<String>> p1 = new ArrayList<>();
        List<List<String>> p2 = new ArrayList<>();

        p1.add(s1);
        p2.add(s1);

        p1.add(s1);
        p2.add(s2);

        p1.add(s1);
        p2.add(s3);

        System.out.println(sim.scores(p1, p2));
    }

    @Test
    public void testInnerProduct() {
        int sentLen = 5;
        int vecLen = 4;
        RealMatrix m = new Array2DRowRealMatrix(sentLen, vecLen);
        RealVector w = new ArrayRealVector(sentLen);
        for (int i = 0; i < sentLen; ++i) {
            double[] col = new double[]{1,2,3,4};
            m.setRow(i, col);
        }
        for (int i = 0; i < sentLen; ++i) {
            w.setEntry(i, 1);
        }
        RealVector res = m.preMultiply(w);
        RealVector exp = new ArrayRealVector(new double[] {5, 10, 15, 20});
        assert res.equals(exp);
    }
}