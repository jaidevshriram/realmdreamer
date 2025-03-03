from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.schedulers import (
    MultiStepSchedulerConfig,
    ExponentialDecaySchedulerConfig,
)

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from realmdreamer.custom_dataparser import CustomDataParserConfig
from realmdreamer.gaussian_splatting.gs_sds_datamanager import GaussianSplattingDatamanagerConfig
from realmdreamer.gaussian_splatting.gs_model import GaussianSplattingModelConfig
from realmdreamer.gaussian_splatting.gs_pipeline import GaussianSplattingPipelineConfig
from realmdreamer.gaussian_splatting.gs_trainer import GaussianSplattingTrainerConfig
from realmdreamer.gaussian_splatting.gs_field import GaussianSplattingFieldConfig

from .dummy_trainer import DummyTrainerConfig
from .dummy_pipeline import DummyPipelineConfig
from .dummy_datamanager import DummyDatamanagerConfig
from .dummy_dataparser import DummyDataParserConfig

dummy_pcd_method = MethodSpecification(
    config=DummyTrainerConfig(
        method_name="dummy-pcd",
        pipeline=DummyPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=DummyDataParserConfig(),
            ),
            model=GaussianSplattingModelConfig(
                gaussian_model=GaussianSplattingFieldConfig(),
                inference_only=True,
                occluded_rand_init=False,
            ),
        ),
        optimizers={},
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Dummy Viewer for Point Clouds",
)
