package com.github.jerinphilip.slimt;

public class Model {
    static {
        System.loadLibrary("slimt_jni");
    }

    public long modelPtr;

    public Model(ModelConfig config, Package archive) {
        modelPtr = Model_createModel(config, archive);
    }

    public void destroy() {
        Model_destroyModel(modelPtr);
    }

    // Native methods
    private native long Model_createModel(ModelConfig config, Package archive);
    private native void Model_destroyModel(long modelPtr);
}
