package com.easemob;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

public class DataIO {

    /* 读取词权重 */
    public Map<String, Double> getWordWeight(String dataDir, String wordFile) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataDir + "/data/" + wordFile)));
        String line;
        Map<String, Double> weightMap = new HashMap<>();
        while ((line = br.readLine()) != null) {
            String[] tmp = line.split("\\s");
            weightMap.put(tmp[0], Double.valueOf(tmp[1]));
        }
        br.close();
        return weightMap;
    }

    public Map<String, Double> getWordWeight(String wordFile) throws IOException {
        return getWordWeight(System.getProperty("user.dir"), wordFile);
    }

    /* 读取glove词向量 */
    public Map<String, Array2DRowRealMatrix> getWordVec(String dataDir, String wordFile) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataDir + "/data/" + wordFile)));
        String line;
        Map<String, Array2DRowRealMatrix> wordMap = new HashMap<>();
        while ((line = br.readLine()) != null) {
            String[] tmp = line.split(" ");
            double[] vecArr = new double[tmp.length - 1];
            for (int i = 1; i < tmp.length; i++) {
                vecArr[i-1] = Double.valueOf(tmp[i]);
            }
            wordMap.put(tmp[0], new Array2DRowRealMatrix(vecArr));
        }
        br.close();
        return wordMap;
    }

    public Map<String, Array2DRowRealMatrix> getWordVec(String wordFile) throws IOException {
        return getWordVec(System.getProperty("user.dir"), wordFile);
    }
}