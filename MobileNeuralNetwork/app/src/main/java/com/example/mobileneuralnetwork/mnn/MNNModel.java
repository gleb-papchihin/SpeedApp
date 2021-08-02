package com.example.mobileneuralnetwork.mnn;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;

public class MNNModel {

    private static final String TAG = MNNModel.class.getName();

    private MNNInstance mInstance;
    private MNNInstance.Session mSession;
    private MNNInstance.Session.Tensor mInputTensor;
    private MNNImageProcess.Config dataConfig;
    private Matrix imgData;
    int inputHeight;
    int inputWidth;
    int numThreads;

    public MNNModel(String modelPath, int width, int height, int threads) throws Exception {
        dataConfig = new MNNImageProcess.Config();
        inputWidth = width;
        inputHeight = height;
        numThreads = threads;
        dataConfig.dest = MNNImageProcess.Format.RGB;
        imgData = new Matrix();

        File file = new File(modelPath);
        if (!file.exists()) {
            throw new Exception("model file is not exists!");
        }

        try {
            mInstance = MNNInstance.createFromFile(modelPath);
            MNNInstance.Config config = new MNNInstance.Config();
            config.numThread = numThreads;
            config.forwardType = MNNForwardType.FORWARD_CPU.type;
            mSession = mInstance.createSession(config);
            mInputTensor = mSession.getInput(null);
        } catch (Exception e) {
            e.printStackTrace();
            throw new Exception("load model fail!");
        }
    }

    public void predictImage(String image_path) throws Exception {
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bitmap = BitmapFactory.decodeStream(fis);
        predictImage(bitmap);
        if (bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    public void predictImage(Bitmap bitmap) throws Exception {
        predict(bitmap);
    }

    private void predict(Bitmap bmp) throws Exception {
        imgData.reset();
        //imgData.postScale(inputWidth / (float) bmp.getWidth(), inputHeight / (float) bmp.getHeight());
        imgData.invert(imgData);
        MNNImageProcess.convertBitmap(bmp, mInputTensor, dataConfig, imgData);

        try {
            mSession.run();
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
    }
}