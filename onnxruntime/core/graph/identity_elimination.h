#pragma once

#include "core/graph/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that eliminates the identity node.
class EliminateIdentity : public RewriteRule {
 public:
  EliminateIdentity() : RewriteRule("EliminateIdentity", "Eliminate identity node") {}

 private:
  bool SatisfyCondition(const Node& node) override;

  Status Apply(GraphEditor* graph_editor, Node* node, bool* modified) override;
};

}  // namespace onnxruntime
