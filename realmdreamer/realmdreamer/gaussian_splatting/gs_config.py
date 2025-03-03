from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (ExponentialDecaySchedulerConfig,
                                          MultiStepSchedulerConfig)
from nerfstudio.plugins.types import MethodSpecification

from realmdreamer.custom_dataparser import CustomDataParserConfig
from realmdreamer.gaussian_splatting.gs_field import \
    GaussianSplattingFieldConfig
from realmdreamer.gaussian_splatting.gs_model import \
    GaussianSplattingModelConfig
from realmdreamer.gaussian_splatting.gs_pipeline import \
    GaussianSplattingPipelineConfig
from realmdreamer.gaussian_splatting.gs_sds_datamanager import \
    GaussianSplattingDatamanagerConfig
from realmdreamer.gaussian_splatting.gs_trainer import \
    GaussianSplattingTrainerConfig
from realmdreamer.optimizers import AdamWOptimizerConfig

realmdreamer = MethodSpecification(
    config=GaussianSplattingTrainerConfig(
        method_name="realmdreamer",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=200,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        pipeline=GaussianSplattingPipelineConfig(
            datamanager=GaussianSplattingDatamanagerConfig(
                dataparser=CustomDataParserConfig(
                    orientation_method="none",
                    center_method="none",
                    auto_scale_poses=False,
                ),
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=GaussianSplattingModelConfig(gaussian_model=GaussianSplattingFieldConfig()),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=5000,
                    lr_pre_warmup=0.01,
                    lr_final=0.000005,
                    max_steps=10000,
                ),
            },
            "f_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
                # "scheduler": MultiStepSchedulerConfig(gamma=0.1, milestones=(2500,)),
            },
            "f_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=7000,
                    lr_pre_warmup=0.005,
                    lr_final=0.0001,
                    max_steps=10000,
                ),
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15, weight_decay=0),
                "scheduler": None,
            },
            "guidance": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=0),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Realmdreamer",
)

realmdreamer_finetune = MethodSpecification(
    config=GaussianSplattingTrainerConfig(
        method_name="realmdreamer-f",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=200,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        pipeline=GaussianSplattingPipelineConfig(
            datamanager=GaussianSplattingDatamanagerConfig(
                dataparser=CustomDataParserConfig(
                    orientation_method="none",
                    center_method="none",
                    auto_scale_poses=False,
                ),
                camera_optimizer=CameraOptimizerConfig(
                    mode="off",
                    optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            model=GaussianSplattingModelConfig(gaussian_model=GaussianSplattingFieldConfig()),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=0.0002, eps=1e-15),
                # "scheduler": None,
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=750,
                    lr_pre_warmup=0.0002,
                    lr_final=0.00000005,
                    max_steps=3000,
                ),
            },
            "f_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "f_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.0001, eps=1e-15),
                "scheduler": None,
                # "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=5000, lr_pre_warmup=0.0001, lr_final=0.00000001, max_steps=15000),
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15, weight_decay=0),
                "scheduler": None,
            },
            "guidance": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15, weight_decay=0),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Realmdreamer for Finetuning",
)
