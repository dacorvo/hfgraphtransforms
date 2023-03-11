import torch
from optimum.fx.optimization.transformations import Transformation


class RemoveDropout(Transformation):
    """
    Transformation that removes Dropout layers.

    This is equivalent to the torch.fx.optimization.remove_dropout method, but having
    a dedicated optimum Transformation allows to compose it with other Transformation.
    """
    preserves_computation = True

    def transform(self, graph_module):
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module = graph_module.get_submodule(node.target)
                if isinstance(module, torch.nn.Dropout) and len(node.args) == 1:
                    # delete dropout from the parent module
                    parent_name, _, name = node.target.rpartition(".")
                    parent_module = graph_module.get_submodule(parent_name)
                    delattr(parent_module, name)
                    # Link all nodes pointing to Dropout to the previous node
                    node.replace_all_uses_with(node.args[0])
                    # Erase node
                    graph_module.graph.erase_node(node)
        return graph_module