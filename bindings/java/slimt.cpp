#include "slimt/slimt.hh"

#include <jni.h>

#include <iostream>
#include <string>
#include <vector>

using namespace slimt;

extern "C" {

JNIEXPORT jlong JNICALL Java_com_example_SlimtService_createService(
    JNIEnv *env, jobject obj, jlong workers, jlong cacheSize) {
  return reinterpret_cast<jlong>(new Service(static_cast<size_t>(workers),
                                             static_cast<size_t>(cacheSize)));
}

JNIEXPORT jobjectArray JNICALL Java_com_example_SlimtService_translate(
    JNIEnv *env, jobject obj, jlong servicePtr, jobjectArray models,
    jobjectArray texts, jboolean html) {
  Service *service = reinterpret_cast<Service *>(servicePtr);
  std::vector<std::string> extractedStrings;

  jsize textsLen = env->GetArrayLength(texts);

  for (int i = 0; i < textsLen; ++i) {
    std::string text = "";  // Convert jstring to std::string
    jobject jtext = env->GetObjectArrayElement(texts, i);
    if (jtext != nullptr) {
      const char *cstr =
          env->GetStringUTFChars(static_cast<jstring>(jtext), nullptr);
      if (cstr != nullptr) {
        text = std::string(cstr);
        env->ReleaseStringUTFChars(static_cast<jstring>(jtext), cstr);
      }
      env->DeleteLocalRef(jtext);
    }

    // Translate text using the service
    Response response = service->translate(std::move(text), html);

    // Extract source and target strings from response and add to vector
    extractedStrings.push_back(response.source.text);
    extractedStrings.push_back(response.target.text);
    // Add other fields from response if needed
  }

  // Convert vector of strings to jobjectArray
  jobjectArray stringArray = env->NewObjectArray(
      extractedStrings.size(), env->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < extractedStrings.size(); ++i) {
    env->SetObjectArrayElement(stringArray, i,
                               env->NewStringUTF(extractedStrings[i].c_str()));
  }

  return stringArray;
}

JNIEXPORT void JNICALL Java_com_example_SlimtService_destroyService(
    JNIEnv *env, jobject obj, jlong servicePtr) {
  delete reinterpret_cast<Service *>(servicePtr);
}

}  // extern "C"
