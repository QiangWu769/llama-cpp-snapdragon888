// Same-process library lifecycle experiment. This is not a CLI product.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-htp-output-pack.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static const char *prompt = "The capital of France is";

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};
struct Batch {
    llama_batch batch;
    explicit Batch(int count) : batch(llama_batch_init(count, 0, 1)) {}
    ~Batch() { llama_batch_free(batch); }
    void set(const std::vector<llama_token> &tokens, int position) {
        batch.n_tokens = int(tokens.size());
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[size_t(i)];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i + 1 == batch.n_tokens;
        }
    }
};
struct HeadEvidence {
    std::array<unsigned, 2> completions {{0, 0}};
    bool rpcmem_weights = true;
};
static bool evaluate(ggml_tensor *tensor, bool ask, void *user) {
    int part = tensor->op == GGML_OP_MUL_MAT && tensor->src[0]
        ? ggml_htp_output_weight_part(tensor->src[0]->name) : -1;
    if (ask) return part >= 0;
    if (part < 0) return true;
    auto &evidence = *static_cast<HeadEvidence *>(user);
    ++evidence.completions[size_t(part)];
    auto *buffer = tensor->src[0]->buffer;
    evidence.rpcmem_weights &= buffer && std::strcmp(
        ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)), "RPCMEM") == 0;
    return true;
}
struct Logits {
    int greedy = -1;
    float minimum = 0;
    float maximum = 0;
    uint64_t hash = UINT64_C(14695981039346656037);
};
struct Iteration {
    int index = 0;
    int prompt_tokens = 0;
    HeadEvidence heads;
    std::array<Logits, 2> logits;
    double load_ms = 0;
    double decode_ms = 0;
    double free_ms = 0;
};
static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static Logits inspect(llama_context *context, int vocabulary) {
    const float *values = llama_get_logits_ith(context, -1);
    if (!values || vocabulary <= 0) throw std::runtime_error("missing logits");
    Logits result;
    result.minimum = std::numeric_limits<float>::infinity();
    result.maximum = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < vocabulary; ++token) {
        float value = values[token];
        if (!std::isfinite(value)) throw std::runtime_error("non-finite logit");
        result.minimum = std::min(result.minimum, value);
        if (value > result.maximum) {
            result.maximum = value;
            result.greedy = token;  // deterministic lowest ID wins ties
        }
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(value), "requires IEEE binary32 size");
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned byte = 0; byte < 4; ++byte) {
            result.hash ^= (bits >> (byte * 8)) & 255;
            result.hash *= UINT64_C(1099511628211);
        }
    }
    return result;
}
static std::vector<llama_token> tokenize(llama_model *model) {
    int length = int(std::strlen(prompt));
    int count = llama_tokenize(model, prompt, length, nullptr, 0, true, true);
    if (count >= 0 || count == std::numeric_limits<int32_t>::min())
        throw std::runtime_error("unexpected tokenizer sizing result");
    std::vector<llama_token> tokens(size_t(-count));
    count = llama_tokenize(model, prompt, length, tokens.data(), int(tokens.size()), true, true);
    if (count <= 0 || count >= 127) throw std::runtime_error("prompt does not fit context");
    tokens.resize(size_t(count));
    return tokens;
}
static Iteration run_iteration(const char *path, int index, std::string &stage) {
    Iteration result;
    result.index = index;
    stage = "load_" + std::to_string(index);
    std::fprintf(stderr, "reload-harness: %s begin\n", stage.c_str());
    auto start = Clock::now();
    auto model_parameters = llama_model_default_params();
    model_parameters.n_gpu_layers = 999;
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(
        llama_load_model_from_file(path, model_parameters), llama_free_model);
    if (!model) throw std::runtime_error("model load failed");
    result.load_ms = elapsed_ms(start);
    auto tokens = tokenize(model.get());
    result.prompt_tokens = int(tokens.size());
    if (llama_n_vocab(model.get()) != 151936)
        throw std::runtime_error("unexpected output vocabulary");
    auto parameters = llama_context_default_params();
    parameters.n_ctx = 128;
    parameters.n_batch = 128;
    parameters.n_ubatch = 32;
    parameters.n_seq_max = 1;
    parameters.n_threads = 4;
    parameters.n_threads_batch = 4;
    parameters.flash_attn = true;
    parameters.cb_eval = evaluate;
    parameters.cb_eval_user_data = &result.heads;
    stage = "context_" + std::to_string(index);
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_new_context_with_model(model.get(), parameters), llama_free);
    if (!context) throw std::runtime_error("context creation failed");
    Batch batch(int(tokens.size()));
    start = Clock::now();
    for (int phase = 0; phase < 2; ++phase) {
        const auto head_counts_before = result.heads.completions;
        stage = std::string(phase ? "decode_" : "prefill_") + std::to_string(index);
        std::fprintf(stderr, "reload-harness: %s begin\n", stage.c_str());
        if (phase == 0) batch.set(tokens, 0);
        else batch.set(std::vector<llama_token>{result.logits[0].greedy}, int(tokens.size()));
        int status = llama_decode(context.get(), batch.batch);
        llama_synchronize(context.get());
        if (status != 0) throw std::runtime_error("decode status " + std::to_string(status));
        if (!result.heads.rpcmem_weights || result.heads.completions[0] == head_counts_before[0] ||
            result.heads.completions[1] == head_counts_before[1]) {
            throw std::runtime_error("both packed RPCMEM heads did not complete this decode");
        }
        result.logits[size_t(phase)] = inspect(context.get(), llama_n_vocab(model.get()));
    }
    result.decode_ms = elapsed_ms(start);
    if (!result.heads.rpcmem_weights || result.heads.completions[0] < 2 || result.heads.completions[1] < 2)
        throw std::runtime_error("both packed RPCMEM heads did not complete both decodes");
    stage = "free_context_" + std::to_string(index);
    std::fprintf(stderr, "reload-harness: %s begin\n", stage.c_str());
    start = Clock::now();
    context.reset();
    stage = "free_model_" + std::to_string(index);
    std::fprintf(stderr, "reload-harness: %s begin\n", stage.c_str());
    model.reset();
    result.free_ms = elapsed_ms(start);
    std::fprintf(stderr, "reload-harness: iteration=%d context/model freed\n", index);
    return result;
}
static std::string quote(const std::string &text) {
    std::ostringstream output;
    output << '"';
    for (unsigned char byte : text) {
        if (byte == '"' || byte == '\\') output << '\\' << char(byte);
        else if (byte < 32) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(byte) << std::dec;
        else output << char(byte);
    }
    output << '"';
    return output.str();
}
static void print(const std::vector<Iteration> &iterations, bool success, bool identical,
                  const std::string &stage, const std::string &error) {
    std::cout << std::setprecision(10) << "{\"success\":" << (success ? "true" : "false")
        << ",\"same_process\":true,\"expected_loads\":2,\"completed_loads\":" << iterations.size()
        << ",\"context_tokens\":128,\"threads\":4,\"flash_attention\":true"
        << ",\"identical_logits_across_reloads\":" << (identical ? "true" : "false")
        << ",\"stage\":" << quote(stage) << ",\"error\":" << (error.empty() ? "null" : quote(error))
        << ",\"iterations\":[";
    for (size_t i = 0; i < iterations.size(); ++i) {
        if (i) std::cout << ',';
        const auto &entry = iterations[i];
        std::cout << "{\"index\":" << entry.index << ",\"prompt_tokens\":" << entry.prompt_tokens
            << ",\"head_completions\":[" << entry.heads.completions[0] << ',' << entry.heads.completions[1]
            << "],\"head_weights_rpcmem\":" << (entry.heads.rpcmem_weights ? "true" : "false")
            << ",\"load_ms\":" << entry.load_ms << ",\"decode_ms\":" << entry.decode_ms
            << ",\"context_and_model_free_ms\":" << entry.free_ms << ",\"logits\":[";
        for (size_t phase = 0; phase < entry.logits.size(); ++phase) {
            if (phase) std::cout << ',';
            const auto &logits = entry.logits[phase];
            std::ostringstream hash;
            hash << std::hex << std::setw(16) << std::setfill('0') << logits.hash;
            std::cout << "{\"phase\":\"" << (phase ? "decode" : "prefill") << "\",\"finite\":true,\"greedy_id\":"
                << logits.greedy << ",\"minimum\":" << logits.minimum << ",\"maximum\":" << logits.maximum
                << ",\"f32_le_fnv1a64\":" << quote(hash.str()) << '}';
        }
        std::cout << "]}";
    }
    std::cout << "],\"mapping_evidence\":\"Both reserved MUL_MAT nodes completed with RPCMEM weights; correlate HTP_TRACE DSP-completed lines for physical execution.\"}\n";
}
int main(int argc, char **argv) {
    std::vector<Iteration> iterations;
    std::string stage = "configuration";
    try {
        if (argc != 2) throw std::runtime_error("usage: hmx-output-reload MODEL.gguf");
        const char *enabled = std::getenv("HTP_OFFLOAD_OUTPUT");
        const char *trace = std::getenv("HTP_TRACE");
        if (!enabled || std::strcmp(enabled, "1") != 0 || !trace || std::strcmp(trace, "1") != 0)
            throw std::runtime_error("requires HTP_OFFLOAD_OUTPUT=1 and HTP_TRACE=1");
        {
            Backend backend;
            for (int i = 0; i < 2; ++i) iterations.push_back(run_iteration(argv[1], i, stage));
            stage = "backend_free";
        }
        bool identical = iterations[0].prompt_tokens == iterations[1].prompt_tokens;
        for (size_t phase = 0; phase < 2; ++phase) {
            const auto &a = iterations[0].logits[phase];
            const auto &b = iterations[1].logits[phase];
            identical &= a.hash == b.hash && a.greedy == b.greedy && a.minimum == b.minimum && a.maximum == b.maximum;
        }
        stage = "complete";
        print(iterations, identical, identical, stage, identical ? "" : "logits changed across reloads");
        return identical ? 0 : 2;
    } catch (const std::exception &exception) {
        print(iterations, false, false, stage, exception.what());
        return 1;
    }
}
