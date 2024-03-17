package com.github.jerinphilip.slimt;

import java.util.List;

public class Service {
  static {
    System.loadLibrary("your_jni_library_name");
  }

  private long servicePtr;

  public Service(long cacheSize) {
    servicePtr = Service_createService(cacheSize);
  }

  public void destroy() {
    Service_destroyService(servicePtr);
  }

  public String[] translate(Model model, List<String> texts, boolean html) {
    return Service_translate(servicePtr, model.modelPtr, texts.toArray(new String[0]), html);
  }

  // Native methods
  private native long Service_createService(long cacheSize);

  private native void Service_destroyService(long servicePtr);

  private native String[] Service_translate(
      long servicePtr, long modelPtr, String[] texts, boolean html);
}
