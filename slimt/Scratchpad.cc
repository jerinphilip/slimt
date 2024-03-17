#include <algorithm>
#include <cstddef>

int main() {
  size_t frontier_size = 0;
  size_t stride = 0;

  for (size_t i = 0; i < frontier_size; i++) {
    auto [logits, attn] = transformer.step(**step);
    for (size_t i = 0; i < stride; i++) {
    std::partial_sort(
		idxs.begin(), 
		idxs.begin() + beam_size, 
		idxs.end(),
            [](const value &
    );
    complete[i] = (words[t] == eos_id)
    }
    if (complete[i]) {
    step.prune(i);
    }
  }
}
