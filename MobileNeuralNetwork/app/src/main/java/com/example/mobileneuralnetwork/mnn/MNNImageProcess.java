package com.example.mobileneuralnetwork.mnn;

import android.graphics.Bitmap;
import android.graphics.Matrix;

public class MNNImageProcess {

    public enum Format {
        /**
         * RGBA
         */
        RGBA(0),
        /**
         * RGB
         */
        RGB(1),
        /**
         * BGR
         */
        BGR(2),
        /**
         * GRAY
         */
        GRAY(3),
        /**
         * BGRA
         */
        BGRA(4),
        /**
         * YUV420
         */
        YUV_420(10),
        /**
         * YUVNV21
         */
        YUV_NV21(11);

        public int type;

        Format(int t) {
            type = t;
        }
    }

    public enum Filter {
        /**
         * NEAREST
         */
        NEAREST(0),
        /**
         * BILINEAL
         */
        BILINEAL(1),
        /**
         * BICUBIC
         */
        BICUBIC(2);

        public int type;

        Filter(int t) {
            type = t;
        }
    }

    public enum Wrap {
        /**
         * CLAMP_TO_EDGE
         */
        CLAMP_TO_EDGE(0),
        /**
         * ZERO
         */
        ZERO(1),
        /**
         * REPEAT
         */
        REPEAT(2);

        public int type;

        Wrap(int t) {
            type = t;
        }
    }


    public static class Config {
        // default
        public float[] mean = {0f, 0f, 0f, 0f};
        public float[] normal = {1f, 1f, 1f, 1f};
        public Format source = Format.RGBA;
        public Format dest = Format.BGR;
        public Filter filter = Filter.NEAREST;
        public Wrap wrap = Wrap.CLAMP_TO_EDGE;
    }

    public static boolean convertBuffer(byte[] buffer, int width, int height, MNNInstance.Session.Tensor tensor, Config config, Matrix matrix) {
        if (matrix == null) {
            matrix = new Matrix();
        }
        float value[] = new float[9];
        matrix.getValues(value);

        return MNNNative.nativeConvertBufferToTensor(buffer, width, height, tensor.instance(),
                config.source.type, config.dest.type, config.filter.type, config.wrap.type, value, config.mean, config.normal);
    }

    public static boolean convertBitmap(Bitmap sourceBitmap, MNNInstance.Session.Tensor tensor, Config config, Matrix matrix) {
        if (matrix == null) {
            matrix = new Matrix();
        }
        float value[] = new float[9];
        matrix.getValues(value);

        return MNNNative.nativeConvertBitmapToTensor(sourceBitmap, tensor.instance(),
                config.dest.type, config.filter.type, config.wrap.type, value, config.mean, config.normal);
    }
}