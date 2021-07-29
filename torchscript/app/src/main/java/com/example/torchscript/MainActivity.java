package com.example.torchscript;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
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

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

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
    private final int       inputHeight     = 64;
    private final int       inputWidth      = 64;
    private final int       inputChannels   = 3;
    private final int       inputBatch      = 1;
    private final String    inputFolder     = "inputs";
    private final String    modelFolder     = "models";
    private final String    modelName       = "yolov5s.ptl";

    // UTILS
    private final DecimalFormat df = new DecimalFormat("#.###");

    // PERMISSIONS
    private static final int PERMISSION_REQUEST_WRITE_STORAGE = 1;

    // ASSETS
    private final String name_external_folder = "/torchscript";
    private final String path_external_assets = Environment.getExternalStorageDirectory().getAbsolutePath() + name_external_folder;

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

    public double estimate_fps() {
        double fps = -1;
        try {
            copyAsset(path_external_assets, modelFolder + "/", modelName);
            Module model = LiteModuleLoader.load(path_external_assets + "/" + modelName);
            AssetManager assetManager = getAssets();
            String[] images_names = assetManager.list(inputFolder);

            // execution time list
            ArrayList<Long> exec_time = new ArrayList<Long>();

            for (String image_name: images_names) {
                Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(inputFolder + "/" + image_name));
                // Resize
                Bitmap resized_bitmap = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true);
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resized_bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                long start = System.currentTimeMillis();
                model.forward(IValue.from(inputTensor));
                long stop = System.currentTimeMillis();

                exec_time.add(stop - start);
            }
            fps = get_mean_fps(exec_time);
        } catch (IOException ioException) {
            Log.d("estimate_fps", ioException.getMessage());
        }
        return fps;
    }

    public void estimate_on_click(View view){
        TextView fps_value = (TextView) findViewById(R.id.fps_value);
        double fps = estimate_fps();
        fps_value.setText(df.format(fps));
    }
}