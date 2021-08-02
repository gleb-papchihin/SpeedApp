package com.example.mobileneuralnetwork;

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

import com.example.mobileneuralnetwork.mnn.MNNModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    // PARAMETERS
    // CUSTOMIZE HERE
    private final int       inputHeight     = 128;
    private final int       inputWidth      = 128;
    private final int       numThreads      = 4;
    private final int       input_repeat    = 8; // Repeat inputs n times.
    private final int       exec_time_shift = 4; // Drop first n elements from exec_time.
    private final String    inputFolder     = "inputs";
    private final String    modelFolder     = "models";
    private final String    modelName       = "yolov5s.mnn";

    // UTILS
    private final DecimalFormat df = new DecimalFormat("#.##");
    private boolean modelWasLoaded = false;

    // PERMISSIONS
    private static final int PERMISSION_REQUEST_WRITE_STORAGE = 1;

    // ASSETS
    private final String name_external_folder = "/mnn";
    private final String path_external_assets = Environment.getExternalStorageDirectory().getAbsolutePath() + name_external_folder;
    private final String modelPath = path_external_assets + "/" + modelName;

    // MOBILE NEURAL NETWORK
    private MNNModel mnnModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Give the app permission to access storage
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_WRITE_STORAGE);
        }
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

    private void copyAsset(String directory, String prefix, String filename) {
        File dir = new File(directory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        AssetManager assetManager = getAssets();
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(prefix + filename);
            File outFile = new File(directory, filename);
            out = new FileOutputStream(outFile);
            copyFile(in, out);
            Toast.makeText(this, "Assets was successfully copied.", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Assets was not copied!", Toast.LENGTH_SHORT).show();
        } finally {
            if (in != null) {
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (out != null) {
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;

        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    public double estimateFps() {
        double fps = -1;
        try {
            // Load model
            if (!modelWasLoaded) {
                mnnModel = new MNNModel(modelPath, inputWidth, inputHeight, numThreads);
            }

            // Init and set list of inputs names.
            AssetManager assetManager = getAssets();
            String[] inputs_names = assetManager.list(inputFolder);

            // Init list of execution time.
            ArrayList<Long> exec_time = new ArrayList<Long>();

            for (int i = 0; i < input_repeat; i++) {
                for (String input_name: inputs_names) {
                    Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(inputFolder + "/" + input_name));
                    Bitmap resized_bitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);
                    long start = System.currentTimeMillis();
                    mnnModel.predictImage(resized_bitmap);
                    long stop = System.currentTimeMillis();

                    exec_time.add(stop - start);
                }
            }
            exec_time = new ArrayList<Long>(exec_time.subList(exec_time_shift, exec_time.size()));
            fps = get_mean_fps(exec_time);

        } catch (Exception exception) {
            Log.d("estimateFps", exception.getMessage());
        }
        return fps;
    }

    public void estimateOnClick(View view) {
        copyAsset(path_external_assets, modelFolder + "/", modelName);
        TextView fps_value = (TextView) findViewById(R.id.fps_value);
        double fps = estimateFps();
        fps_value.setText(df.format(fps));
    }
}