package com.easemob;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import java.util.Map;

public class DataIOTest {

    private DataIO data = new DataIO();

    @Test
    public void testWordMap() throws Exception {
        String wordFile = "glove.test.txt";
        Map<String, Array2DRowRealMatrix> wordMap = data.getWordVec(wordFile);
        assert wordMap.keySet().size() > 0;
    }

    @Test
    public void testWeightMap() throws Exception {
        String wordFile = "idf.txt";
        Map<String, Double> weightMap = data.getWordWeight(wordFile);
        assert weightMap.keySet().size() > 0;
    }
}