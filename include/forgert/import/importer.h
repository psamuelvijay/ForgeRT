#pragma once

/**
 * @file importer.h
 * @brief Build a ForgeRT Graph from a parsed .fgx model (ONNX opset 7 subset).
 *
 * The importer owns the operator and attribute allowlist. Anything outside it
 * fails with ImportError naming the node index, node name and op type; there is
 * no fallback and no partially built model.
 *
 * Supported ONNX ops (opset 7): Conv, Relu, Concat, MaxPool, AveragePool,
 * Dropout, Reshape.
 *
 * Weights: v1 has no graph constants, so every float32 initializer becomes an
 * extra graph input registered after the model's own inputs. ImportedModel
 * owns those tensors and appends them on every run(). The only int64 tensor
 * accepted is a Reshape shape initializer, which is folded into a fixed-target
 * ReshapeOp after an int32 range check.
 *
 * Shapes: every node output is compared exactly against the converter's shape
 * inference results (VINF) and the declared graph outputs (OUTP).
 */

#include "forgert/graph/graph.h"
#include "forgert/import/fgx.h"
#include "forgert/tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace forgert {

class ImportError : public std::runtime_error {
public:
    explicit ImportError(const std::string& message)
        : std::runtime_error("import: " + message) {}
};

/// Range-checked int64 -> int32 conversion. Throws ImportError naming `what`
/// and the offending index if any value is outside [INT32_MIN, INT32_MAX].
std::vector<int32_t> narrowInt64ToInt32(const std::vector<int64_t>& values,
                                        const std::string& what);

class ImportedModel {
public:
    /// Build from a parsed model. Throws ImportError (or fgx::FormatError via load()).
    static ImportedModel fromFgx(const fgx::Model& model);

    /// fgx::readFile + fromFgx.
    static ImportedModel load(const std::string& path);

    ImportedModel(ImportedModel&&) = default;
    ImportedModel& operator=(ImportedModel&&) = default;

    const Graph& graph() const { return graph_; }

    /// Model inputs supplied by the caller (excludes weights).
    const std::vector<std::string>& inputNames()  const { return input_names_; }
    const std::vector<TensorShape>& inputShapes() const { return input_shapes_; }
    const std::vector<std::string>& outputNames() const { return output_names_; }

    /// Number of float32 initializers registered as extra graph inputs.
    size_t numWeights() const { return weights_.size(); }

    /// Non-fatal notes, e.g. a node output with no VINF entry to check against.
    const std::vector<std::string>& warnings() const { return warnings_; }

    /// Run on CPU. `inputs` must match inputNames()/inputShapes() exactly
    /// (Float32, CPU). Returns outputs in outputNames() order.
    std::vector<std::unique_ptr<Tensor>> run(const std::vector<const Tensor*>& inputs);

private:
    ImportedModel() = default;

    Graph graph_;
    std::vector<std::unique_ptr<Tensor>> weights_;
    std::vector<std::string> input_names_;
    std::vector<TensorShape> input_shapes_;
    std::vector<std::string> output_names_;
    std::vector<std::string> warnings_;
};

} // namespace forgert