
import sys
from ..utils import (
    _LazyModule,
)

_import_structure = {
    "sparsity_util": ["sparsify_matrix_for_FC_layer", "mask_attention_result", "find_layers"],
    "wrapper_layer": ["WrapperLayer"],
    "instrumentation_context": ["InstrumentationContext"],
}

sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)