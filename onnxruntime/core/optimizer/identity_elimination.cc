// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/op.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/identity_elimination.h"

namespace onnxruntime {

Status EliminateIdentity::Apply(Graph& graph, Node& node, bool& modified, bool& deleted) {
  if (utils::RemoveSingleInSingleOutNode(graph, node)) {
    modified = deleted = true;
  }

  return Status::OK();
}

bool EliminateIdentity::SatisfyCondition(const Node& node) {
  return utils::IsSingleInSingleOutNode(node);
}

}  // namespace onnxruntime
