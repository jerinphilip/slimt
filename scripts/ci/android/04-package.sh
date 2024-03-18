#!/bin/bash

cd bindings/java

# Compile Java files
javac com/github/jerinphilip/slimt/*.java

# Package Java class files into JAR
jar cf slimt.jar com/github/jerinphilip/slimt/*.class

# Structure your JAR file
mkdir -p com/github/jerinphilip/slimt/jni/android-arm64-v8a

# Copy compiled JNI libraries into the JAR
# Package everything into final JAR file
cp ../../build/bindings/java/libslimt_jni.so com/github/jerinphilip/slimt/jni/android-arm64-v8a/libslimt_jni.so
jar uf slimt.jar com/github/jerinphilip/slimt/jni/android-arm64-v8a/libslimt_jni.so
