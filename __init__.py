import os, sys, importlib.util

# Normalize package name so relative imports work even if loaded by absolute path
if __name__ != "MagicNodes":
    sys.modules["MagicNodes"] = sys.modules[__name__]
    __package__ = "MagicNodes"
    # Precreate subpackage alias MagicNodes.mod
    _mod_pkg_name = "MagicNodes.mod"
    _mod_pkg_dir = os.path.join(os.path.dirname(__file__), "mod")
    _mod_pkg_file = os.path.join(_mod_pkg_dir, "__init__.py")
    if _mod_pkg_name not in sys.modules and os.path.isfile(_mod_pkg_file):
        _spec = importlib.util.spec_from_file_location(
            _mod_pkg_name, _mod_pkg_file, submodule_search_locations=[_mod_pkg_dir]
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_mod_pkg_name] = _mod
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)

# Imports of active nodes
from .mod.mg_combinode import MG_CombiNode
from .mod.mg_upscale_module import MG_UpscaleModule
from .mod.mg_adaptive import MG_AdaptiveSamplerHelper
from .mod.mg_cade import MG_ComfyAdaptiveDetailEnhancer, MG_CADEOptions
from .mod.mg_ids import MG_IntelligentDetailStabilizer
from .mod.mg_seed_latent import MG_SeedLatent
from .mod.mg_sagpu_attention import MG_SagpuAttention
from .mod.mg_cleanup import MG_CleanUp
from .mod.mg_latent_adapter import MG_LatentAdapter
from .mod.mg_controlfusion import MG_ControlFusion, MG_ControlFusionOptions
from .mod.mg_zesmart_sampler import MG_ZeSmartScheduler


NODE_CLASS_MAPPINGS = {
    "MG_AdaptiveSamplerHelper": MG_AdaptiveSamplerHelper,
    "MG_CombiNode": MG_CombiNode,
    "MG_SeedLatent": MG_SeedLatent,
    "PatchSageAttention": MG_SagpuAttention,
    "MG_CleanUp": MG_CleanUp,
    "MG_LatentAdapter": MG_LatentAdapter,  # experimental
    "MG_UpscaleModule": MG_UpscaleModule,
    "MG_CADE": MG_ComfyAdaptiveDetailEnhancer,
    "MG_IntelligentDetailStabilizer": MG_IntelligentDetailStabilizer,
    "MG_ControlFusion": MG_ControlFusion,
    "MG_ZeSmartSampler": MG_ZeSmartScheduler,
    "MG_CADEOptions": MG_CADEOptions,
    "MG_ControlFusionOptions": MG_ControlFusionOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombiNode": "MG_CombiNode",
    "Seed Latent": "MG_SeedLatent",
    "Patch Sage Attention": "PatchSageAttention",
    "Clean Up": "MG_CleanUp",
    "Latent Adapter": "MG_LatentAdapter",  # experimental
    "CADE": "MG_CADE",
    "Control Fusion": "MG_ControlFusion",
    "ZeSmartSampler": "MG_ZeSmartSampler",
    "Intelligent Detail Stabilizer": "MG_IDS",
    "Upscale Module": "MG_UpscaleModule",
    "Control Fusion Options": "MG_ControlFusionOptions",
    "CADE Options": "MG_CADEOptions",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
