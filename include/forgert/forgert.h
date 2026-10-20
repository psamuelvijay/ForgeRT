#pragma once

/**
 * @brief ForgeRT - Resource-Aware Neural Inference Runtime
 *
 * Top-level convenience header. Includes all public APIs.
 *
 * Usage:
 *   #include "forgert/forgert.h"
 *   using namespace forgert;
 */

// Tensor abstractions
#include "forgert/tensor/device.h"
#include "forgert/tensor/dtype.h"
#include "forgert/tensor/shape.h"
#include "forgert/tensor/tensor.h"

// Memory (Phase 4)
#include "forgert/memory/memory_stats.h"

// Operators
#include "forgert/operator/operator.h"
#include "forgert/operator/add.h"
#include "forgert/operator/relu.h"
#include "forgert/operator/matmul.h"
#include "forgert/operator/softmax.h"
#include "forgert/operator/layernorm.h"
#include "forgert/operator/conv2d.h"
#include "forgert/operator/concat.h"
#include "forgert/operator/maxpool2d.h"
#include "forgert/operator/averagepool2d.h"
#include "forgert/operator/dropout.h"
#include "forgert/operator/reshape.h"

// Graph
#include "forgert/graph/graph.h"

// Graph optimizer (Phase 5)
#include "forgert/graph/optimizer.h"

// Resource-aware scheduler (Phase 6)
#include "forgert/scheduler/scheduler.h"
