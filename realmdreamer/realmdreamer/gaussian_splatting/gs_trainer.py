from __future__ import annotations

import dataclasses
import functools
import os
import pdb
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (check_eval_enabled, check_main_thread,
                                         check_viewer_enabled)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

# from nerfstudio.viewer.server.viewer_state import ViewerState
# from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str

# os.environ['HF_HUB_OFFLINE']='1'


@dataclass
class GaussianSplattingTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: GaussianSplattingTrainer)


class GaussianSplattingTrainer(Trainer):

    def setup(self) -> None:
        super().setup()

        self.pipeline.config.max_num_iterations = self.config.max_num_iterations

    @profiler.time_function
    def train_iteration(self, step: int, get_image_dict: bool) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"
        for _ in range(self.gradient_accumulation_steps):

            start_iter = time.time()
            model_outputs, loss_dict, metrics_dict, image_dict, stats_dict = self.pipeline.get_train_loss_dict(
                step=step, get_image_dict=get_image_dict
            )
            loss = functools.reduce(torch.add, loss_dict.values())
            loss /= self.gradient_accumulation_steps
            end_iter = time.time()
            # print(f"Time taken for iteration: {end_iter - start_iter}")

            loss.backward()

        # max_norm = 50
        # torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), max_norm)
        self.optimizers.optimizer_step_all()

        # Post optimizer step function call
        densification_counts = self.pipeline.post_optimizer_step(step, model_outputs, self.optimizers)
        stats_dict.update(densification_counts)

        # total_grad = 0
        # for tag, value in self.pipeline.model.named_parameters():
        #     assert tag != "Total"
        #     if value.grad is not None and ('field' in tag or 'proposal' in tag or 'Total' in tag):
        #         grad = value.grad.norm()
        #         metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
        #         total_grad += grad

        # metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict, image_dict, stats_dict  # type: ignore

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        torch.set_float32_matmul_precision("medium")

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        curr_step = 0

        # Save the code that is used
        self.save_code()
        self.save_checkpoint(curr_step)

        self._init_viewer_state()

        # Initialise the model - inpaint all holes somehow just for initialization purposes
        # self.pipeline.inpaint_all_holes()

        try:
            with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
                num_iterations = self.config.max_num_iterations

                # reset optimizer again just before training (mainly for case where you load from a checkpoint)
                self.pipeline.model.gaussian_model.reload_optimizer(self.optimizers.optimizers)

                # with torch.autograd.detect_anomaly():
                with open("/tmp/a.txt", "w") as f:

                    for step in range(self._start_step, self._start_step + num_iterations):

                        curr_step = step

                        with self.train_lock:
                            with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                                self.pipeline.train()

                                # training callbacks before the training iteration
                                for callback in self.callbacks:
                                    callback.run_callback_at_location(
                                        step,
                                        location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION,
                                    )

                                # time the forward pass
                                get_image_dict = step_check(
                                    step,
                                    self.config.logging.steps_per_log,
                                    run_at_zero=True,
                                )
                                (
                                    loss,
                                    loss_dict,
                                    metrics_dict,
                                    image_dict,
                                    stats_dict,
                                ) = self.train_iteration(step, get_image_dict=get_image_dict)

                                # training callbacks after the training iteration
                                for callback in self.callbacks:
                                    callback.run_callback_at_location(
                                        step,
                                        location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION,
                                    )

                        # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                        if step > 1:
                            writer.put_time(
                                name=EventName.TRAIN_RAYS_PER_SEC,
                                duration=self.world_size
                                * self.pipeline.datamanager.get_train_rays_per_batch()
                                / max(0.001, train_t.duration),
                                step=step,
                                avg_over_steps=True,
                            )

                        self._update_viewer_state(step)

                        # a batch of train rays
                        if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                            writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                            writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                            writer.put_dict(
                                name="Train Metrics Dict",
                                scalar_dict=metrics_dict,
                                step=step,
                            )
                            writer.put_dict(
                                name="Train Stats Dict",
                                scalar_dict=stats_dict,
                                step=step,
                            )

                            # Write Histograms
                            # for tag, value in self.pipeline.model.named_parameters():
                            #     writer.put_histogram(name=f"Values/{tag}", values=value, step=step)

                            # Write images
                            group = "Train Images"
                            for key, image in image_dict.items():
                                writer.put_image(name=group + "/" + key, image=image, step=step)

                            # The actual memory allocated by Pytorch. This is likely less than the amount
                            # shown in nvidia-smi since some unused memory can be held by the caching
                            # allocator and some context needs to be created on GPU. See Memory management
                            # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                            # for more details about GPU memory management.
                            writer.put_scalar(
                                name="GPU Memory (MB)",
                                scalar=torch.cuda.max_memory_allocated() / (1024**2),
                                step=step,
                            )

                        # # Do not perform evaluation if there are no validation images
                        # if self.pipeline.datamanager.eval_dataset:
                        #     self.eval_iteration(step)

                        if step_check(step, self.config.steps_per_save):
                            self.save_checkpoint(step)

                        writer.write_out_storage()
        except Exception as e:

            # save checkpoint on error
            self.save_checkpoint(curr_step)
            raise e

        # save checkpoint at the end of training
        self.save_checkpoint(curr_step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(
            Panel(
                table,
                title="[bold][green]:tada: Training Finished :tada:[/bold]",
                expand=False,
            )
        )

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            # self._start_step = loaded_state["step"] + 1

            self._start_step = 1
            loaded_state["step"] = 1

            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            # self.optimizers.load_optimizers(loaded_state["optimizers"])

            # if 'schedulers' in loaded_state:
            # self.optimizers.load_schedulers(loaded_state["schedulers"])

            # Reset params for the optimizer
            self.pipeline.reset_params(self.optimizers.optimizers)

            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1

            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            # self.optimizers.load_optimizers(loaded_state["optimizers"])

            # if 'schedulers' in loaded_state:
            # self.optimizers.load_schedulers(loaded_state["schedulers"])

            # Reset params for the optimizer
            self.pipeline.reset_params(self.optimizers.optimizers)

            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"

        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "guidance" not in k}

        torch.save(
            {
                "step": step,
                "pipeline": (
                    self.pipeline.module.state_dict()  # type: ignore
                    if hasattr(self.pipeline, "module")
                    else pipeline_state_dict
                ),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )

        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    # Save the code that is used
    def save_code(self) -> None:

        root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Copy all files in the root folder recursively to the checkpoint directory under `code` folder
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith(".py"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(
                        self.checkpoint_dir,
                        "..",
                        "code",
                        src_path.replace(root_folder, "").strip("/"),
                    )
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    os.system(f"cp {src_path} {dst_path}")
