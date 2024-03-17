package com.github.jerinphilip.slimt;

import java.util.List;

public class Service {
  static {
    System.loadLibrary("slimt_jni");
  }

  private long servicePtr;

  public Service(long cacheSize) {
    servicePtr = ncreate(cacheSize);
  }

  public void destroy() {
    ndestroy(servicePtr);
  }

  public String[] translate(Model model, List<String> texts, boolean html) {
    return ntranslate(servicePtr, model.modelPtr, texts.toArray(new String[0]), html);
  }

  // Native methods
  private native long ncreate(long cacheSize);

  private native void ndestroy(long servicePtr);

  private native String[] ntranslate(long servicePtr, long modelPtr, String[] texts, boolean html);
}
