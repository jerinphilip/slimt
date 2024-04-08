#include "slimt/slimt.hh"

#include <jni.h>

#include <iostream>
#include <string>
#include <vector>

using namespace slimt;  // NOLINT

// using Service = Async;
using Service = Blocking;

extern "C" {

// NOLINTBEGIN
// Model
#define SLIMT_JNI_EXPORT(cls, method) \
  JNICALL Java_io_github_jerinphilip_slimt_##cls##_##method

JNIEXPORT jlong SLIMT_JNI_EXPORT(Model, ncreate)(JNIEnv *env, jobject obj,
                                                 jobject jconfig,
                                                 jobject jpackage) {
  // Extract Config object fields
  jclass cls = env->GetObjectClass(jconfig);
  jfieldID encoder_layers_field = env->GetFieldID(cls, "encoder_layers", "J");
  jfieldID decoder_layers_field = env->GetFieldID(cls, "decoder_layers", "J");
  jfieldID ffn_depth_field = env->GetFieldID(cls, "feed_forward_depth", "J");
  jfieldID num_heads_field = env->GetFieldID(cls, "num_heads", "J");
  jfieldID split_mode_field =
      env->GetFieldID(cls, "split_mode", "Ljava/lang/String;");

  jlong j_encoder_layers = env->GetLongField(jconfig, encoder_layers_field);
  jlong j_decoder_layers = env->GetLongField(jconfig, decoder_layers_field);
  jlong j_ffn_depth = env->GetLongField(jconfig, ffn_depth_field);
  jlong j_num_heads = env->GetLongField(jconfig, num_heads_field);
  jstring j_split_mode =
      (jstring)env->GetObjectField(jconfig, split_mode_field);
  const char *split_mode_cstr = env->GetStringUTFChars(j_split_mode, NULL);

  // Create Config object
  slimt::Model::Config config;
  config.encoder_layers = static_cast<size_t>(j_encoder_layers);
  config.decoder_layers = static_cast<size_t>(j_decoder_layers);
  config.feed_forward_depth = static_cast<size_t>(j_ffn_depth);
  config.num_heads = static_cast<size_t>(j_num_heads);
  config.split_mode = std::string(split_mode_cstr);

  // Extract Package object fields
  // Assuming Package object contains necessary fields for Model creation

  jclass package_cls = env->GetObjectClass(jpackage);
  jfieldID model_field =
      env->GetFieldID(package_cls, "model", "Ljava/lang/String;");
  jfieldID vocabulary_field =
      env->GetFieldID(package_cls, "vocabulary", "Ljava/lang/String;");
  jfieldID shortlist_field =
      env->GetFieldID(package_cls, "shortlist", "Ljava/lang/String;");
  jfieldID ssplit_field =
      env->GetFieldID(package_cls, "ssplit", "Ljava/lang/String;");

  jstring j_model = (jstring)env->GetObjectField(jpackage, model_field);
  jstring j_vocabulary =
      (jstring)env->GetObjectField(jpackage, vocabulary_field);
  jstring j_shortlist = (jstring)env->GetObjectField(jpackage, shortlist_field);
  jstring j_ssplit = (jstring)env->GetObjectField(jpackage, ssplit_field);

  const char *model_cstr = env->GetStringUTFChars(j_model, nullptr);
  const char *vocabulary_cstr = env->GetStringUTFChars(j_vocabulary, nullptr);
  const char *shortlist_cstr = env->GetStringUTFChars(j_shortlist, nullptr);
  const char *ssplit_cstr = env->GetStringUTFChars(j_ssplit, nullptr);

  // Create Package object
  slimt::Package<std::string> package;
  package.model = std::string(model_cstr);
  package.vocabulary = std::string(vocabulary_cstr);
  package.shortlist = std::string(shortlist_cstr);
  package.ssplit = std::string(ssplit_cstr);

  // Release Java string references
  env->ReleaseStringUTFChars(j_model, model_cstr);
  env->ReleaseStringUTFChars(j_vocabulary, vocabulary_cstr);
  env->ReleaseStringUTFChars(j_shortlist, shortlist_cstr);
  env->ReleaseStringUTFChars(j_ssplit, ssplit_cstr);

  // Create Model object
  slimt::Model *model = new slimt::Model(config, package);

  // Clean up
  env->ReleaseStringUTFChars(j_split_mode, split_mode_cstr);

  return reinterpret_cast<jlong>(model);
}

JNIEXPORT void SLIMT_JNI_EXPORT(Model, ndestroy)(JNIEnv *env, jobject obj,
                                                 jlong model_addr) {
  delete reinterpret_cast<Model *>(model_addr);
}

// Service
JNIEXPORT jlong SLIMT_JNI_EXPORT(Service, ncreate)(JNIEnv *env, jobject obj,
                                                   jlong cache_size) {
  Config config;
  config.cache_size = cache_size;
  return reinterpret_cast<jlong>(new Service(config));
}

JNIEXPORT void SLIMT_JNI_EXPORT(Service, ndestroy)(JNIEnv *env, jobject obj,
                                                   jlong service_addr) {
  delete reinterpret_cast<Service *>(service_addr);
}

JNIEXPORT jobjectArray SLIMT_JNI_EXPORT(Service, ntranslate)(
    JNIEnv *env, jobject obj, jlong service_addr, jobject jmodel,
    jobjectArray texts, jboolean html) {
  Service *service = reinterpret_cast<Service *>(service_addr);
  std::vector<std::string> sources;
  std::vector<std::string> targets;

  jsize length = env->GetArrayLength(texts);

  for (int i = 0; i < length; ++i) {
    std::string text = "";  // Convert jstring to std::string
    jobject jtext = env->GetObjectArrayElement(texts, i);
    if (jtext != nullptr) {
      const char *cstr =
          env->GetStringUTFChars(static_cast<jstring>(jtext), nullptr);
      if (cstr != nullptr) {
        text = std::string(cstr);
        sources.push_back(text);
        env->ReleaseStringUTFChars(static_cast<jstring>(jtext), cstr);
      }
      env->DeleteLocalRef(jtext);
    }
  }

  // Translate text using the service
  Model *model_raw_ptr = reinterpret_cast<Model *>(jmodel);
  auto pseudo_deleter = [](Model *model_raw_ptr) {};
  Ptr<Model> model(model_raw_ptr, pseudo_deleter);
  Options options{
      .html = static_cast<bool>(html)  //
  };

  Responses responses = service->translate(model, std::move(sources), options);
  for (Response &response : responses) {
    targets.push_back(response.target.text);
  }

  // Convert vector of strings to jobjectArray
  jobjectArray jtargets = env->NewObjectArray(
      targets.size(), env->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < targets.size(); ++i) {
    env->SetObjectArrayElement(jtargets, i,
                               env->NewStringUTF(targets[i].c_str()));
  }

  return jtargets;
}

// NOLINTEND

#undef SLIMT_JNI_EXPORT

}  // extern "C"
