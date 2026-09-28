#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: bitnet-prefill-control <pinned-gguf> [prompt]\n");
        return 1;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    const auto load_start = std::chrono::steady_clock::now();
    llama_model* model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load native BitNet control GGUF\n");
        llama_backend_free();
        return 2;
    }
    const auto load_end = std::chrono::steady_clock::now();

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_seq_max = 1;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    llama_context* context = llama_init_from_model(model, context_params);
    if (!context) {
        std::fprintf(stderr, "failed to initialize native BitNet context\n");
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const char* prompt = argc == 3 ? argv[2]
        : "State: No temperature reading is available. Choose: A yes, B no, C unknown. Answer:";
    const std::size_t prompt_length = std::strlen(prompt);
    if (prompt_length == 0 || prompt_length > 2048) {
        std::fprintf(stderr, "native BitNet control prompt must contain 1-2048 bytes\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }
    const int32_t prompt_bytes = static_cast<int32_t>(prompt_length);
    const int32_t needed = llama_tokenize(vocab, prompt, prompt_bytes, nullptr, 0, true, false);
    if (needed >= 0 || -needed > 128) {
        std::fprintf(stderr, "unexpected native BitNet prompt token count\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(-needed));
    const int32_t count = llama_tokenize(
        vocab, prompt, prompt_bytes, tokens.data(), -needed, true, false
    );
    if (count != -needed || count <= 0) {
        std::fprintf(stderr, "native BitNet prompt tokenization failed\n");
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    const std::array<const char*, 3> labels = {"A", "B", "C"};
    std::array<llama_token, 3> label_tokens;
    for (std::size_t option = 0; option < labels.size(); ++option) {
        const std::string candidate = std::string(prompt) + labels[option];
        const int32_t required = llama_tokenize(
            vocab, candidate.c_str(), static_cast<int32_t>(candidate.size()),
            nullptr, 0, true, false
        );
        std::vector<llama_token> candidate_tokens(static_cast<std::size_t>(count + 1));
        const int32_t candidate_count = required == -(count + 1) ? llama_tokenize(
            vocab, candidate.c_str(), static_cast<int32_t>(candidate.size()),
            candidate_tokens.data(), count + 1, true, false
        ) : -1;
        if (
            candidate_count != count + 1
            || !std::equal(tokens.begin(), tokens.end(), candidate_tokens.begin())
            || llama_vocab_is_control(vocab, candidate_tokens.back())
            || llama_vocab_is_eog(vocab, candidate_tokens.back())
            || std::find(label_tokens.begin(), label_tokens.begin() + option,
                         candidate_tokens.back()) != label_tokens.begin() + option
        ) {
            std::fprintf(stderr, "native BitNet answer label is not a distinct one-token continuation\n");
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 7;
        }
        label_tokens[option] = candidate_tokens.back();
    }

    llama_batch batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int32_t index = 0; index < count; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = (index == count - 1);
    }
    const auto prefill_start = std::chrono::steady_clock::now();
    const int32_t status = llama_decode(context, batch);
    const auto prefill_end = std::chrono::steady_clock::now();
    const float* logits = status == 0 ? llama_get_logits_ith(context, count - 1) : nullptr;
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    int32_t finite_logits = 0;
    int32_t best_token = -1;
    float best_logit = -INFINITY;
    if (logits && vocabulary_size > 0) {
        for (int32_t token = 0; token < vocabulary_size; ++token) {
            if (std::isfinite(logits[token])) {
                ++finite_logits;
                if (logits[token] > best_logit) {
                    best_logit = logits[token];
                    best_token = token;
                }
            }
        }
    }
    const bool valid_logits = status == 0 && finite_logits == vocabulary_size && best_token >= 0;
    if (valid_logits) {
        std::array<float, 3> option_logits;
        float option_max = -INFINITY;
        for (std::size_t option = 0; option < labels.size(); ++option) {
            option_logits[option] = logits[label_tokens[option]];
            option_max = std::max(option_max, option_logits[option]);
        }
        double option_sum = 0.0;
        double vocabulary_sum = 0.0;
        for (float value : option_logits) {
            option_sum += std::exp(static_cast<double>(value - option_max));
        }
        for (int32_t token = 0; token < vocabulary_size; ++token) {
            vocabulary_sum += std::exp(static_cast<double>(logits[token] - best_logit));
        }
        const int selected = static_cast<int>(
            std::max_element(option_logits.begin(), option_logits.end()) - option_logits.begin()
        );
        const double allowed_mass = option_sum * std::exp(
            static_cast<double>(option_max - best_logit)
        ) / vocabulary_sum;
        const auto milliseconds = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        std::printf(
            "{\"backend\":\"pinned_bitnet_control\",\"prompt_tokens\":%d,"
            "\"vocab_size\":%d,\"finite_logits\":%d,\"argmax_token_id\":%d,"
            "\"generated_tokens\":0,\"labels\":[\"A\",\"B\",\"C\"],"
            "\"label_token_ids\":[%d,%d,%d],\"conditional_probabilities\":[%.9f,%.9f,%.9f],"
            "\"selected_label\":\"%s\",\"max_option_probability\":%.9f,"
            "\"allowed_label_mass\":%.9f,\"calibration_status\":\"uncalibrated\","
            "\"load_ms\":%.3f,\"prefill_ms\":%.3f}\n",
            count, vocabulary_size, finite_logits, best_token,
            label_tokens[0], label_tokens[1], label_tokens[2],
            std::exp(static_cast<double>(option_logits[0] - option_max)) / option_sum,
            std::exp(static_cast<double>(option_logits[1] - option_max)) / option_sum,
            std::exp(static_cast<double>(option_logits[2] - option_max)) / option_sum,
            labels[selected],
            std::exp(static_cast<double>(option_logits[selected] - option_max)) / option_sum,
            allowed_mass,
            milliseconds(load_end - load_start), milliseconds(prefill_end - prefill_start)
        );
    } else {
        std::fprintf(stderr, "native BitNet prefill/logits failed: %d\n", status);
    }
    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return valid_logits ? 0 : 6;
}