// forgert_run: run a single-input, single-output .fgx model on CPU.
//
//   forgert_run model.fgx input.bin output.bin
//
// input.bin holds exactly the model input's elements as raw little-endian
// float32 in C order; output.bin receives the output the same way.
// Prints the import summary, shapes, and wall-clock times to stderr.
// Exit codes: 0 ok, 1 usage/IO, 2 model rejected, 3 inference error.
#include "forgert/import/importer.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace forgert;

static std::string shapeStr(const TensorShape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.ndim(); ++i) out += (i ? "," : "") + std::to_string(s.dim(i));
    return out + "]";
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: forgert_run model.fgx input.bin output.bin\n";
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };

    const auto t0 = clock::now();
    std::unique_ptr<ImportedModel> model;
    try {
        model = std::make_unique<ImportedModel>(ImportedModel::load(argv[1]));
    } catch (const std::exception& e) {
        std::cerr << "forgert_run: " << e.what() << "\n";
        return 2;
    }
    const auto t1 = clock::now();
    for (const auto& w : model->warnings()) std::cerr << "warning: " << w << "\n";
    if (model->inputNames().size() != 1 || model->outputNames().size() != 1) {
        std::cerr << "forgert_run: only single-input, single-output models are supported\n";
        return 2;
    }

    const TensorShape& in_shape = model->inputShapes()[0];
    Tensor input(in_shape, DataType::Float32, Device::CPU);
    std::error_code ec;
    const auto size = std::filesystem::file_size(argv[2], ec);
    if (ec || size != input.numBytes()) {
        std::cerr << "forgert_run: " << argv[2] << " must be exactly " << input.numBytes()
                  << " bytes (float32 " << shapeStr(in_shape) << ")"
                  << (ec ? "" : ", got " + std::to_string(size)) << "\n";
        return 1;
    }
    {
        std::ifstream f(argv[2], std::ios::binary);
        f.read(static_cast<char*>(input.data()), static_cast<std::streamsize>(input.numBytes()));
        if (!f) { std::cerr << "forgert_run: short read on " << argv[2] << "\n"; return 1; }
    }

    std::vector<std::unique_ptr<Tensor>> outputs;
    const auto t2 = clock::now();
    try {
        outputs = model->run({&input});
    } catch (const std::exception& e) {
        std::cerr << "forgert_run: inference failed: " << e.what() << "\n";
        return 3;
    }
    const auto t3 = clock::now();

    const Tensor& out = *outputs[0];
    std::ofstream f(argv[3], std::ios::binary);
    f.write(static_cast<const char*>(out.data()), static_cast<std::streamsize>(out.numBytes()));
    if (!f) { std::cerr << "forgert_run: cannot write " << argv[3] << "\n"; return 1; }

    std::cerr << "model: " << model->graph().numNodes() << " nodes, " << model->numWeights()
              << " weight tensors; import " << ms(t1 - t0) << " ms\n"
              << "input  " << model->inputNames()[0] << " " << shapeStr(in_shape) << "\n"
              << "output " << model->outputNames()[0] << " " << shapeStr(out.shape()) << "\n"
              << "inference (CPU) " << ms(t3 - t2) << " ms\n";
    return 0;
}