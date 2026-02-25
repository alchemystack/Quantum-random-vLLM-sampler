"""Token selection subsystem for qr-sampler.

CDF-based token selection driven by an external uniform random value.
Implements temperature scaling, top-k, softmax, top-p, and CDF binary search.
"""

from qr_sampler.selection.selector import TokenSelector
from qr_sampler.selection.types import SelectionResult

__all__ = [
    "SelectionResult",
    "TokenSelector",
]
