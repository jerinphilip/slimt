package com.github.jerinphilip.slimt;

public class Model {
    static {
        System.loadLibrary("slimt_jni");
    }

    private long modelPtr;

    public Model(ModelConfig config, Package<String> package) {
        modelPtr = Model_createModel(config, package);
    }

    public void destroy() {
        Model_destroyModel(modelPtr);
    }

    // Native methods
    private native long Model_createModel(ModelConfig config, Package<String> package);
    private native void Model_destroyModel(long modelPtr);
}
