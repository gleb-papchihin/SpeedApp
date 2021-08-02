package com.example.mobileneuralnetwork.mnn;

import android.util.Log;

public class MNNInstance {
    private static final String TAG = MNNInstance.class.getSimpleName();

    public static MNNInstance createFromFile(String fileName) {
        long instance = MNNNative.nativeCreateNetFromFile(fileName);
        if (0 == instance) {
            Log.e(TAG, "Create Net Failed from file " + fileName);
            return null;
        }

        return new MNNInstance(instance);
    }


    public static class Config {
        public int forwardType = MNNForwardType.FORWARD_CPU.type;
        public int numThread = 4;
        public String[] saveTensors = null;
        public String[] outputTensors = null;

    }

    public class Session {
        public class Tensor {
            private Tensor(long ptr) {
                mTensorInstance = ptr;
            }

            protected long instance() {
                return mTensorInstance;
            }

            public void reshape(int[] dims) {
                MNNNative.nativeReshapeTensor(mNetInstance, mTensorInstance, dims);
                mData = null;
            }

            public void setInputIntData(int[] data) {
                MNNNative.nativeSetInputIntData(mNetInstance, mTensorInstance, data);
                mData = null;
            }

            public void setInputFloatData(float[] data) {
                MNNNative.nativeSetInputFloatData(mNetInstance, mTensorInstance, data);
                mData = null;
            }

            public int[] getDimensions() {
                return MNNNative.nativeTensorGetDimensions(mTensorInstance);
            }

            public float[] getFloatData() {
                getData();
                return mData;
            }

            public int[] getIntData() {
                if (null == mIntData) {
                    int size = MNNNative.nativeTensorGetIntData(mTensorInstance, null);
                    mIntData = new int[size];
                }
                MNNNative.nativeTensorGetIntData(mTensorInstance, mIntData);

                return mIntData;
            }

            public void getData() {
                if (null == mData) {
                    int size = MNNNative.nativeTensorGetData(mTensorInstance, null);
                    mData = new float[size];
                }
                MNNNative.nativeTensorGetData(mTensorInstance, mData);
            }

            public byte[] getUINT8Data() {
                if (null == mUINT8Data) {
                    int size = MNNNative.nativeTensorGetUINT8Data(mTensorInstance, null);
                    mUINT8Data = new byte[size];
                }
                MNNNative.nativeTensorGetUINT8Data(mTensorInstance, mUINT8Data);

                return mUINT8Data;
            }

            private float[] mData = null;
            private int[] mIntData = null;
            private byte[] mUINT8Data = null;
            private long mTensorInstance;
        }


        private Session(long ptr) {
            mSessionInstance = ptr;
        }

        //After all input tensors' reshape, call this method
        public void reshape() {
            MNNNative.nativeReshapeSession(mNetInstance, mSessionInstance);
        }

        public void run() {
            MNNNative.nativeRunSession(mNetInstance, mSessionInstance);
        }

        public Tensor[] runWithCallback(String[] names) {
            long[] tensorPtr = new long[names.length];

            Tensor[] tensorReturnArray = new Tensor[names.length];
            MNNNative.nativeRunSessionWithCallback(mNetInstance, mSessionInstance, names, tensorPtr);
            for (int i = 0; i < names.length; i++) {
                tensorReturnArray[i] = new Tensor(tensorPtr[i]);
            }
            return tensorReturnArray;
        }

        public Tensor getInput(String name) {
            long tensorPtr = MNNNative.nativeGetSessionInput(mNetInstance, mSessionInstance, name);
            if (0 == tensorPtr) {
                Log.e(TAG, "Can't find seesion input: " + name);
                return null;
            }
            return new Tensor(tensorPtr);
        }

        public Tensor getOutput(String name) {
            long tensorPtr = MNNNative.nativeGetSessionOutput(mNetInstance, mSessionInstance, name);
            if (0 == tensorPtr) {
                Log.e(TAG, "Can't find seesion output: " + name);
                return null;
            }
            return new Tensor(tensorPtr);
        }

        //Release the session from net instance, it's not needed if you call net.release()
        public void release() {
            checkValid();
            MNNNative.nativeReleaseSession(mNetInstance, mSessionInstance);
            mSessionInstance = 0;
        }

        private long mSessionInstance = 0;
    }

    public Session createSession(Config config) {
        checkValid();

        if (null == config) {
            config = new Config();
        }

        long sessionId = MNNNative.nativeCreateSession(mNetInstance, config.forwardType, config.numThread, config.saveTensors, config.outputTensors);
        if (0 == sessionId) {
            Log.e(TAG, "Create Session Error");
            return null;
        }
        return new Session(sessionId);
    }

    private void checkValid() {
        if (mNetInstance == 0) {
            throw new RuntimeException("MNNNetInstance native pointer is null, it may has been released");
        }
    }


    public void release() {
        checkValid();
        MNNNative.nativeReleaseNet(mNetInstance);
        mNetInstance = 0;
    }

    private MNNInstance(long instance) {
        mNetInstance = instance;
    }

    private long mNetInstance;
}