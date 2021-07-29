package com.example.tflite;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    // PARAMETERS
    // CUSTOMIZE HERE
    private final int       inputBatch      = 1;
    private final int       inputChannels   = 3;
    private final int       inputHeight     = 128;
    private final int       inputWidth      = 128;
    private final int[]     output_shape    = {1, 1008, 85};
    private final String    modelPath       = "models/yolov5s-fp16.tflite";
    private final int       threads         = 2;
    private final DataType  modelDtype      = DataType.FLOAT32;

    // ASSETS
    private final String inputFolder = "inputs";

    // UTILS
    private final DecimalFormat df = new DecimalFormat("#.###");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public double get_mean_seconds(ArrayList<Long> milliseconds) {
        long sum = 0;
        for (long millisecond: milliseconds) {
            sum += millisecond;
        }
        double mean_m = (double) sum / (double) milliseconds.size();
        return mean_m / 1000;
    }

    public double get_mean_fps(ArrayList<Long> milliseconds) {
        double mean_s = get_mean_seconds(milliseconds);
        double epsilon = 0.0000001;
        double fps = -1;
        if (mean_s > epsilon) {
            fps = 1 / mean_s;
        }
        return fps;
    }

    public  double estimate_fps() {
        double fps = -1;
        try {

            // Load model file to byte buffer
            Context context = getApplicationContext();
            MappedByteBuffer mappedByteBuffer = FileUtil.loadMappedFile(context, modelPath);

            // Create interpreter
            Interpreter.Options options = new Interpreter.Options();

            // Set number of threads
            options.setNumThreads(threads);

            // Setup interpreter
            Interpreter interpreter = new Interpreter(mappedByteBuffer, options);

            AssetManager assetManager = getAssets();
            String[] images_names = assetManager.list(inputFolder);

            // Image pre-processor
            ImageProcessor imageProcessor = new ImageProcessor.Builder().add(
                    new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR)
            ).build();

            // execution time list
            ArrayList<Long> exec_time = new ArrayList<Long>();

            for (String image_name: images_names) {

                TensorBuffer outputs = TensorBuffer.createFixedSize( output_shape, modelDtype );

                // Load image
                TensorImage tensorImage = new TensorImage(modelDtype);
                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(
                        inputFolder + "/" + image_name));

                // Pre-process loaded image
                tensorImage.load(bitmap);
                tensorImage = imageProcessor.process(tensorImage);

                // inference
                long start = System.currentTimeMillis();
                interpreter.run(tensorImage.getBuffer(), outputs.getBuffer());
                long stop = System.currentTimeMillis();

                exec_time.add(stop - start);
            }
            return get_mean_fps(exec_time);
        } catch (IOException ioException) {
            Log.d("estimate_on_click", ioException.getMessage());
        }
        return fps;
    }

    public void estimate_on_click(View view) {
        TextView fps_value = (TextView) findViewById(R.id.fps_value);
        double fps = estimate_fps();
        fps_value.setText(df.format(fps));
    }
}