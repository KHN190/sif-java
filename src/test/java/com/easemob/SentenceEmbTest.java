package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SentenceEmbTest {

    private SentenceEmb emb = new SentenceEmb();

    public SentenceEmbTest() throws IOException {
    }

    @Test
    public void testWeightedAvg() throws Exception {
        List<String> text = new ArrayList<>();
        text.add("good");
        text.add("morning");
        System.out.println(emb.weightedAvg(text));
    }

    @Test
    public void testEmb() throws Exception {
        List<String> text = new ArrayList<>();
        text.add("good");
        text.add("morning");
        RealMatrix e1 = emb.embedding(text);

        text.add("mike");
        RealMatrix e2 = emb.embedding(text);

        assert e1.getColumnDimension() == e2.getColumnDimension();
        assert e1.getRowDimension() == e1.getRowDimension();
    }

    @Test
    public void testMatrixEmb() throws Exception {
        List<List<String>> texts = new ArrayList<>();
        List<String> t1 = new ArrayList<>();
        List<String> t2 = new ArrayList<>();

        t1.add("good");
        t1.add("morning");

        t2.add("hello");
        t2.add("world");

        texts.add(t1);
        texts.add(t2);

        RealMatrix m = emb.matrixEmbedding(texts);
        assert(m.getRowDimension() == 2);
        assert(m.getColumnDimension() == 50);
    }}