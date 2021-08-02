package com.example.onnx;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {

    // PARAMETERS
    // CUSTOMIZE HERE
    private final int       inputBatch      = 1;
    private final int       inputChannels   = 3;
    private final int       inputWidth      = 64;
    private final int       inputHeight     = 64;
    private final int       input_repeat    = 8; // Repeat inputs n times.
    private final int       exec_time_shift = 4; // Drop first n elements from exec_time.
    private final String    modelPath       = "models/yolov5s.all.ort";
    private final boolean   torchMode       = true;

    // ONNX PARAMETERS
    OrtSession.SessionOptions.OptLevel optLevel = OrtSession.SessionOptions.OptLevel.ALL_OPT;
    OrtSession.SessionOptions.ExecutionMode executionMode = OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL;

    // ASSETS
    private final String inputFolder = "inputs";

    // UTILS
    private final DecimalFormat df = new DecimalFormat("#.###");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    private void fillInputArrayTorchMode(Bitmap bitmap, float[][][][] inputArray) {
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int pixel = bitmap.getPixel(x, y);
                inputArray[0][0][y][x] = ((pixel >> 16) & 0xFF);
                inputArray[0][1][y][x] = ((pixel >> 8) & 0xFF);
                inputArray[0][2][y][x] = ((pixel) & 0xFF);
            }
        }
    }

    private void fillInputArrayTensorflowMode(Bitmap bitmap, float[][][][] inputArray) {
        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int pixel = bitmap.getPixel(x, y);
                inputArray[0][y][x][0] = ((pixel >> 16) & 0xFF);
                inputArray[0][y][x][1] = ((pixel >> 8) & 0xFF);
                inputArray[0][y][x][2] = ((pixel) & 0xFF);
            }
        }
    }

    private Bitmap loadResizedInput(String inputPath, int resizedWidth, int resizedHeight) throws IOException{
        Bitmap bitmap = BitmapFactory.decodeStream(
                getAssets().open(inputPath));
        Bitmap resized_bitmap = Bitmap.createScaledBitmap(bitmap,
                resizedWidth, resizedHeight, true);
        return bitmap;
    }

    private byte[] loadModelAsBytes(String modelPath) throws IOException{
        InputStream inputStream = getAssets().open(modelPath);
        int size = inputStream.available();
        byte[] buffer = new byte[size];
        inputStream.read(buffer);
        inputStream.close();
        return buffer;
    }

    public double getMeanSeconds(ArrayList<Long> milliseconds) {
        long sum = 0;
        for (long millisecond: milliseconds) {
            sum += millisecond;
        }
        double meanMill = (double) sum / (double) milliseconds.size();
        return meanMill / 1000;
    }

    public double getMeanFps(ArrayList<Long> milliseconds) {
        double meanSec = getMeanSeconds(milliseconds);
        double epsilon = 0.0000001;
        double fps = -1;
        if (meanSec > epsilon) {
            fps = 1 / meanSec;
        }
        return fps;
    }

    private double estimateFpsTorchMode(String modelPath, int numThreads,
                                OrtSession.SessionOptions.OptLevel graphOptimizationLevel,
                                OrtSession.SessionOptions.ExecutionMode executionMode) {

        ArrayList<Long> exec_time = new ArrayList<Long>();

        try {
            byte[]      modelBytes  = loadModelAsBytes(modelPath);
            String[]    inputsNames = getAssets().list(inputFolder);

            try {

                // Session options does not include to default onnxruntime-mobile building.
                // Create a custom building if you wanna optimize the code.

                //SessionOptions sessionOptions = new SessionOptions();

                // Set optimization level: [OptLevel.NO_OPT, OptLevel.ALL_OPT]
                //sessionOptions.setOptimizationLevel(graphOptimizationLevel);

                // Set number of threads: [integer]
                //sessionOptions.setInterOpNumThreads(numThreads);

                // Set execution mode: [ExecutionMode.PARALLEL, ExecutionMode.SEQUENTIAL]
                //sessionOptions.setExecutionMode(executionMode);

                // Set an environment
                OrtEnvironment env = OrtEnvironment.getEnvironment();

                // Create session from model bytes
                OrtSession session = env.createSession(modelBytes);

                // Load input meta
                Map<String, NodeInfo> inputMetaMap = session.getInputInfo();
                NodeInfo inputMeta = inputMetaMap.values().iterator().next();
                for (int i = 0; i < input_repeat; i++) {
                    for (String inputName : inputsNames) {

                        // Load input as bitmap
                        Bitmap bitmap = loadResizedInput(inputFolder + "/" + inputName, inputWidth, inputHeight);

                        // Create and fill float array
                        float[][][][] inputArray = new float[inputBatch][inputChannels][inputHeight][inputWidth];
                        fillInputArrayTorchMode(bitmap, inputArray);

                        // Create onnx input tensor
                        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray);
                        Map<String, OnnxTensor> container = new HashMap<>();
                        container.put(inputMeta.getName(), inputTensor);

                        long start = System.currentTimeMillis();
                        OrtSession.Result result = session.run(container);
                        long stop = System.currentTimeMillis();
                        Log.d("Time", "" + (stop - start));
                        exec_time.add(stop - start);
                    }
                }
                exec_time = new ArrayList<Long>(exec_time.subList(exec_time_shift, exec_time.size()));

            } catch (OrtException ortException) {
                Log.d("EstimateFPS: ORTEx", ortException.getMessage());
            }

        } catch (IOException ioException) {
            Log.d("EstimateFPS: IOEx", ioException.getMessage());
        }

        return getMeanFps(exec_time);
    }

    private double estimateFpsTensorflowMode(String modelPath, int numThreads,
                                        OrtSession.SessionOptions.OptLevel graphOptimizationLevel,
                                        OrtSession.SessionOptions.ExecutionMode executionMode) {

        ArrayList<Long> exec_time = new ArrayList<Long>();

        try {
            byte[]      modelBytes  = loadModelAsBytes(modelPath);
            String[]    inputsNames = getAssets().list(inputFolder);

            try {

                // Session options does not include to default onnxruntime-mobile building.
                // Create a custom building if you wanna optimize the code.

                //SessionOptions sessionOptions = new SessionOptions();

                // Set optimization level: [OptLevel.NO_OPT, OptLevel.ALL_OPT]
                //sessionOptions.setOptimizationLevel(graphOptimizationLevel);

                // Set number of threads: [integer]
                //sessionOptions.setInterOpNumThreads(numThreads);

                // Set execution mode: [ExecutionMode.PARALLEL, ExecutionMode.SEQUENTIAL]
                //sessionOptions.setExecutionMode(executionMode);

                // Set an environment
                OrtEnvironment env = OrtEnvironment.getEnvironment();

                // Create session from model bytes
                OrtSession session = env.createSession(modelBytes);

                // Load input meta
                Map<String, NodeInfo> inputMetaMap = session.getInputInfo();
                NodeInfo inputMeta = inputMetaMap.values().iterator().next();

                for (String inputName : inputsNames) {

                    // Load input as bitmap
                    Bitmap bitmap = loadResizedInput(inputFolder + "/" + inputName, inputWidth, inputHeight);

                    // Create and fill float array
                    float[][][][] inputArray = new float[inputBatch][inputChannels][inputHeight][inputWidth];
                    fillInputArrayTensorflowMode(bitmap, inputArray);

                    // Create onnx input tensor
                    OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray);
                    Map<String, OnnxTensor> container = new HashMap<>();
                    container.put(inputMeta.getName(), inputTensor);

                    long start = System.currentTimeMillis();
                    OrtSession.Result result = session.run(container);
                    long stop = System.currentTimeMillis();
                    Log.d("Time", "" + (stop-start));
                    exec_time.add(stop - start);
                }

            } catch (OrtException ortException) {
                Log.d("EstimateFPS: ORTEx", ortException.getMessage());
            }

        } catch (IOException ioException) {
            Log.d("EstimateFPS: IOEx", ioException.getMessage());
        }

        return getMeanFps(exec_time);
    }

    public void estimateFpsOnClick(View view) {
        double fps = 0;
        TextView fps_value_core_1 = (TextView) findViewById(R.id.fps_value_core_1);
        if (torchMode) {
            fps = estimateFpsTorchMode(modelPath, 1, optLevel, executionMode);
            fps_value_core_1.setText(df.format(fps));
        } else {
            fps = estimateFpsTensorflowMode(modelPath, 1, optLevel, executionMode);
            fps_value_core_1.setText(df.format(fps));
        }
    }
}