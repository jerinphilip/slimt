package com.github.jerinphilip.slimt;

public class Model {
  static {
    System.loadLibrary("slimt_jni");
  }

  public long modelPtr;

  public Model(ModelConfig config, Package archive) {
    modelPtr = ncreate(config, archive);
  }

  public void destroy() {
    ndestroy(modelPtr);
  }

  // Native methods
  private native long ncreate(ModelConfig config, Package archive);

  private native void ndestroy(long modelPtr);
}
