package com.example.nativespeedapp;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.example.nativespeedapp.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    // MAIN PARAMETERS
    String saved_assets_path    = Environment.getExternalStorageDirectory().getAbsolutePath() + "/speedapp_files";
    String model_name           = "YoloV5S";
    boolean torch_input_mode    = true;

    // PATHS TO MODELS
    String path_torch           = saved_assets_path + "/yolov5s.torchscript.pt";
    String path_ort             = saved_assets_path + "/yolov5s.all.ort";
    String path_tflite          = saved_assets_path + "/yolov5s.tflite";

    // INPUT SETTINGS
    int width                   = 640;
    int height                  = 640;
    int n_channels              = 3;

    // PATHS TO IMAGES
    String saved_images_path    = saved_assets_path + "/images/";
    String[] names_images       = {
            "1.jpg", "2.jpg", "3.jpg", "4.jpg",
            "5.jpg", "6.jpg", "7.jpg", "8.jpg",
            "9.jpg", "10.jpg", "11.jpg"
    };

    // Elements
    TextView tflite_fps_text_view;
    TextView torch_fps_text_view;
    TextView ort_fps_text_view;


    private static final int MY_PERMISSION_REQUEST_STORAGE = 1;
    private ActivityMainBinding binding;
    Button estimate;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Give the app permission to access storage
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_PERMISSION_REQUEST_STORAGE);
        }

        // Copy assets folder to an external storage.
        copyAssets();

        // Init view elements
        estimate = (Button) findViewById(R.id.estimate_button);
        tflite_fps_text_view = (TextView) findViewById(R.id.tflite_fps);
        torch_fps_text_view = (TextView) findViewById(R.id.torch_fps);
        ort_fps_text_view = (TextView) findViewById(R.id.ort_fps);
        DecimalFormat df = new DecimalFormat("#.####");

        // Handler for framework estimation
        estimate.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String path_torch = saved_assets_path + "/yolov5s.torchscript.pt";
                String path_tflite = saved_assets_path + "/yolov5s.tflite";
                String path_ort = saved_assets_path + "/yolov5s.all.ort";

                String paths_images = combine_strings_with_separator(names_images,
                        saved_images_path, ";");


                double ort_fps = estimate_ort_fps(path_torch, paths_images,
                        height, width, n_channels, torch_input_mode);
                ort_fps_text_view.setText(df.format(ort_fps));
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                double tflite_fps = estimate_tflite_fps(path_tflite, paths_images,
                        height, width, n_channels, torch_input_mode);
                tflite_fps_text_view.setText(df.format(tflite_fps));
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                double torch_fps = estimate_torch_fps(path_torch, paths_images,
                        height, width, n_channels);
                torch_fps_text_view.setText(df.format(torch_fps));
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case MY_PERMISSION_REQUEST_STORAGE: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    if (ContextCompat.checkSelfPermission(MainActivity.this,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                        Toast.makeText(this, "Permission was not granted!", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        }
    }

    private String combine_strings_with_separator(String[] strings, String prefix, String separator) {
        StringBuilder combined = new StringBuilder();
        for (String string: strings) {
            combined.append(prefix).append(string).append(separator);
        }
        return combined.toString();
    }

    private void copyAssets() {
        AssetManager assetManager = getAssets();
        String[] files = null;
        try {
            files = assetManager.list("");
            for (String filename: files) {
                copyAsset(filename);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void copyAsset(String filename) {
        String dirPath = saved_assets_path;
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        AssetManager assetManager = getAssets();
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(filename);
            File outFile = new File(dirPath, filename);
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

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native double estimate_tflite_fps(String path_model, String paths_images, int height,
                                             int width, int n_channels, boolean torch_input_mode);
    public native double estimate_torch_fps(String path_model, String paths_images, int height,
                                            int width, int n_channels);
    public native double estimate_ort_fps(String path_model, String paths_images, int height,
                                          int width, int n_channels, boolean torch_input_mode);

}