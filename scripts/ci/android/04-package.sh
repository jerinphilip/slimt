#!/bin/bash

cd bindings/java

# Compile Java files
javac io/github/jerinphilip/slimt/*.java

# Package Java class files into JAR
jar cf slimt.jar io/github/jerinphilip/slimt/*.class

# Structure your JAR file
mkdir -p io/github/jerinphilip/slimt/jni/android-arm64-v8a

# Copy iopiled JNI libraries into the JAR
# Package everything into final JAR file
cp ../../build/bindings/java/libslimt_jni.so io/github/jerinphilip/slimt/jni/android-arm64-v8a/libslimt_jni.so
jar uf slimt.jar io/github/jerinphilip/slimt/jni/android-arm64-v8a/libslimt_jni.so
