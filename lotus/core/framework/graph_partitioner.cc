#include "core/framework/graph_partitioner.h"

#include "core/framework/kernel_registry_manager.h"
#include "core/graph/function.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/execution_providers.h"

using namespace ::Lotus::Common;
namespace Lotus {
Status GraphPartitioner::Partition(LotusIR::Graph& graph) const {
  if (providers_.Empty()) {
    return Status(LOTUS, INVALID_ARGUMENT, "No provider specified.");
  }

  // Partitioning <graph> based on provider preference and their capabilities.
  auto kernel_registries = kernel_registry_mgr_.GetAllKernelRegistries();
  for (auto& provider : providers_) {
    auto capability_results = provider->GetCapability(graph, kernel_registries);
    int count = 0;
    for (auto& sub_graph : capability_results) {
      if (nullptr == sub_graph) {
        continue;
      }
      if (1 == sub_graph->nodes.size()) {
        // The <provider> can run a single node in the <graph>.
        auto node = graph.GetNode(sub_graph->nodes[0]);
        if (nullptr != node && node->GetExecutionProviderType().empty()) {
          node->SetExecutionProviderType(provider->Type());
        }
      } else {
        // The <provider> can run a fused <sub_graph> in the <graph>.
        //
        // Add fused node into <graph>
        LOTUS_ENFORCE(nullptr != sub_graph->GetMetaDef());
        std::string node_name = provider->Type() + "_" + sub_graph->GetMetaDef()->name + "_" + std::to_string(count++);
        auto fused_node = graph.FuseSubGraph(std::move(sub_graph), node_name);
        fused_node->SetExecutionProviderType(provider->Type());
      }
    }
  }

  for (auto& node : graph.Nodes()) {
    if (!graph.IsSourceNode(node) && !graph.IsSinkNode(node) && node.GetExecutionProviderType().empty()) {
      return Status(LOTUS, FAIL, "Partitioning failed. No execution provider is capable of running node (" + node.Name() + ").");
    }
  }

  return Status::OK();
}
}  // namespace Lotus
