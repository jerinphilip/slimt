#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace detail {

inline bool file_exists(const std::string &fpath) {
  return (access(fpath.c_str(), F_OK) == 0);
}

inline void write_to_bin(const std::string &fpath, uint8_t *data, size_t size) {
  size_t count = 1;
  FILE *fp = fopen(fpath.c_str(), "wb");  // w for write, b for binary
  fwrite(data, size, count, fp);          // write 10 bytes from our buffer
}

inline void variable_name_transform(std::string &buffer) {
  std::replace(buffer.begin(), buffer.end(), ' ', '_');
  std::replace(buffer.begin(), buffer.end(), '[', '_');
  std::replace(buffer.begin(), buffer.end(), ']', '_');
  std::replace(buffer.begin(), buffer.end(), '=', '_');
}

template <class NodeType>
std::string var_metadata(NodeType node) {
  std::stringstream stream;
  stream << node->value_type() << "_";
  stream << node->shape();
  if (node->name() != "none") {
    std::string name = node->name();

    size_t lead = 4;
    if (name.size() > lead && name.substr(0, lead) == "F0::") {
      name = name.substr(lead, std::string::npos);  // NOLINT
    }
    stream << "_" << name;
  }
  std::string identifier = stream.str();

  variable_name_transform(identifier);
  return identifier;
}

inline std::string extract_op_name(const char *pretty_fn) {
  // Extract function name.
  const char *start = pretty_fn + std::char_traits<char>::length("marian::");
  const char *end = start;
  while (*end != '\0' && *end != ':') {
    end++;
  }

  std::string name(start, end - start);
  return name;
}

//  Save only for unique problem-sizes, at least at the beginning.
template <class Node>
std::string save_to_disk(const std::string &name, Node node) {
  const char *env_save_path = std::getenv("DEBUG_VARIABLES_SAVE_PATH");
  if (env_save_path) {
    std::string save_path(env_save_path);
    std::string abs_path = save_path + "/" + name;
    // std::cerr << "Trying to save " << abs_path << "\n";
    if (not file_exists(abs_path)) {
      auto memory = node->val()->memory();
      write_to_bin(abs_path, memory->data(), memory->size());
      // std::cerr << "Saved " << abs_path << "\n";
      return abs_path;
    }
  }

  return "";
}

template <class NodeType>
inline void var_id(std::ostream &out, NodeType value) {
  out << "var_" << value->getId() << " ";
  out << value->value_type() << " ";
  out << "[" << value->shape() << "]";
  if (value->name() != "none") {
    out << " " << value->name();
  }
}

template <class NodeType>
inline bool process(const char *pretty_fn, NodeType *value, std::ostream &out) {
  std::stringstream stream;
  std::string op_name = extract_op_name(pretty_fn);
  std::string lhs_tag = var_metadata(value);
  std::string prefix =
      "var_" + std::to_string(value->getId()) + "-" + op_name + "-" + lhs_tag;
  std::string lhs_name = prefix + "-lhs.bin";
  std::string lhs_save = save_to_disk(lhs_name, value);

  stream << "after: ";
  var_id(stream, value);
  if (!lhs_save.empty()) {
    stream << " " << lhs_name;
  }

  auto children = value->children();
  if (not children.empty()) {
    stream << "\noperands: \n";
  }
  for (size_t i = 0; i < children.size(); i++) {
    auto rhs = children[i];
    stream << "  - ";
    var_id(stream, rhs);

    std::string rhs_tag = var_metadata(rhs);
    // NOLINTBEGIN
    std::string rhs_name =
        prefix + "-rhs" + std::to_string(i) + "-" + rhs_tag + ".bin";
    // NOLINTEND
    std::string rhs_save = save_to_disk(rhs_name, rhs);
    if (!rhs_save.empty()) {
      stream << " " << rhs_name;
    }
    stream << "\n";
  }

  out << stream.str();
  return not lhs_save.empty();
}
}  // namespace detail

#define THREAD_GUARD(body) \
  [&]() {                  \
    body;                  \
  }()  // test if THREAD_GUARD is neccessary, remove if no problems occur.
       //
#if 1
#define NodeOp(op)                                                  \
  [=]() {                                                           \
    std::stringstream stream;                                       \
    stream << "file: \"" << __FILE__ << "\"\n";                     \
    stream << "line: " << __LINE__ << "\n";                         \
    stream << "fn: \"" << __PRETTY_FUNCTION__ << "\"\n";            \
    stream << "op: { " << #op << " }\n";                            \
    stream << "before: ";                                           \
    detail::var_id(stream, this);                                   \
    op;                                                             \
    stream << "\n";                                                 \
    bool flag = detail::process(__PRETTY_FUNCTION__, this, stream); \
    stream << "\n\n";                                               \
    if (flag) {                                                     \
      std::cerr << stream.str();                                    \
    };                                                              \
  }
#else
#define NodeOp(op) [=]() { op; }
#endif
