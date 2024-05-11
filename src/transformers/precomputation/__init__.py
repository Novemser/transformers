
import sys
from ..utils import (
    _LazyModule,
)

_import_structure = {
    "training_data_generator": ["generate_act_hats_llama2"]
}

sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)