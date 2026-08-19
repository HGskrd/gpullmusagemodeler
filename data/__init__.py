"""Public compatibility surface for the GPU/LLM planner catalog."""

# Re-export order mirrors catalog construction: models are assembled before quality
# and capability overlays replace their final public entries.
# ruff: noqa: F403,I001

from .specs import *
from .hardware import *
from .model_class import *
from .asr_support import *
from .embedding_support import *
from .text_support import *
from .models import *
from .quality import *
from .presets import *
from .cloud import *
from .use_cases import *
from .environment import *
from .groups import *
