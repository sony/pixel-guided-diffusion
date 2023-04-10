from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf

from .guidance import GuidanceConfig
from nnabla_diffusion.config.python.datasetddpm import DatasetDDPMConfig


@dataclass
class RuntimeConfig:
    device_id: str = "0"


@dataclass
class SegmentationGuidanceConfig:
    runtime: RuntimeConfig = MISSING
    datasetddpm: DatasetDDPMConfig = MISSING
    seg_guide: GuidanceConfig = MISSING
