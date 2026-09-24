# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate curriculum-guided finance tasks, train Snowball, and evaluate on FinanceBench.

The ``tasks`` stage needs an Iris client, the GLM relay, and ``GLM_BULK_TOKEN`` in its
environment. Preview the full dependency graph with::

    uv run python -m experiments.post_training.curriculum_sft.snowball_pipeline --version 2026.09.24

Add ``--run`` to execute it, or ``--stage tasks`` to build only the generation step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import click
from fray.cluster import ResourceConfig
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.model_config import ModelConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.models import models
from experiments.evaluation.pipeline import eval_checkpoint_step
from experiments.models import ModelConfig as DownloadModelConfig
from experiments.models import download_model
from experiments.post_training.curriculum_sft.data import STORE_RELATIVE_PATH, FinanceStore, dataset_step, store_step
from experiments.post_training.curriculum_sft.generation import ACCEPTED_EXAMPLES, generation_step
from experiments.sft.launcher import PreparedModel, SFTSpec

SNOWBALL_TOKENIZER = "marin-community/marin-tokenizer"
SNOWBALL_EOS_TOKEN_IDS = (128001, 128009)
SNOWBALL_EVALUATION_MODEL = "snowball-datakit-sft-2026-09-20"
FINANCEBENCH_CONFIG = Path("experiments/evaluation/configs/evalchemy/financebench.yaml")
COREWEAVE_CLUSTER = "cw-rno2a"
COREWEAVE_PREFIX = "s3://marin-us-east-02a/marin"
TEMP_TTL_DAYS = 7
TRAIN_STEPS = 1
TRAIN_BATCH_SIZE = 64
TRAIN_SEQUENCE_LENGTH = 4096
DATA_AXIS_SIZE = 8
EXPERT_AXIS_SIZE = 8
_TRAIN_RESOURCES = "train_resources"


def _base_model() -> ModelConfig:
    model = models()[SNOWBALL_EVALUATION_MODEL]
    if model.revision is None:
        raise ValueError(f"{SNOWBALL_EVALUATION_MODEL} must pin an immutable revision")
    return model


def _temporary_path(version: str, name: str) -> str:
    return marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/{version}/{name}",
        source_prefix=COREWEAVE_PREFIX,
    )


def _staged_model() -> ArtifactStep[LevanterCheckpoint]:
    model = _base_model()
    step = download_model(DownloadModelConfig(hf_repo_id=model.location, hf_revision=model.revision))
    output = marin_temp_bucket(
        TEMP_TTL_DAYS,
        prefix=f"curriculum-sft/snowball/base-hf/{model.revision}",
        source_prefix=COREWEAVE_PREFIX,
    )
    return dataclasses.replace(step, override_path=output)


def _sft_spec(staged_model: ArtifactStep[LevanterCheckpoint], version: str) -> SFTSpec:
    return SFTSpec(
        name=user_owned_name("checkpoints/curriculum-sft/snowball/finance"),
        version=version,
        model=PreparedModel(
            step=staged_model,
            model_type="snowball",
            eos_token_ids=SNOWBALL_EOS_TOKEN_IDS,
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=(),
        optimizer=AdamConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1.0,
            weight_decay=0.0,
            lr_schedule="constant",
            warmup=0.0,
            min_lr_ratio=0.0,
        ),
        mesh=MeshConfig(
            axes={"expert": EXPERT_AXIS_SIZE, "replica": 1, "model": 1},
            dcn_axes={"data": DATA_AXIS_SIZE, "replica_dcn": 1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        ),
        seq_len=TRAIN_SEQUENCE_LENGTH,
        pack=True,
        batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=TRAIN_STEPS,
        wandb_project="marin-curriculum-sft-snowball",
    )


def _training_data(cache_path: str, tokenizer: str) -> LmDataConfig:
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={
            "finance": DatasetComponent(
                source=None,
                cache_dir=cache_path,
                format=TextLmDatasetFormat(),
                pack=True,
            )
        },
        train_weights={"finance": 1.0},
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )


def _sft_step(
    store: ArtifactStep[FinanceStore],
    staged_model: ArtifactStep[LevanterCheckpoint],
    *,
    version: str,
) -> ArtifactStep[LevanterCheckpoint]:
    spec = _sft_spec(staged_model, version)
    source = spec.model

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        tokenizer = source.resolve_tokenizer(ctx)
        data = _training_data(prefix_join(ctx.artifact_path(store), STORE_RELATIVE_PATH), tokenizer)
        pod_config = source.build_train_config(ctx, spec, data, ctx.runtime_arg(_TRAIN_RESOURCES), TRAIN_STEPS)
        train_config = dataclasses.replace(pod_config.train_config, z_loss_weight=1e-4, hf_save_dtype="bfloat16")
        return dataclasses.replace(pod_config, train_config=train_config)

    return ArtifactStep(
        name=spec.name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=source.run,
        build_config=build_config,
        deps=(store, *source.init_deps()),
        runtime_args={
            _TRAIN_RESOURCES: ResourceConfig.with_gpu(
                "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=8, preemptible=False
            )
        },
        override_path=_temporary_path(version, "sft"),
    )


def build_pipeline(version: str) -> dict[str, ArtifactStep]:
    """Build the single finance task → data → Snowball SFT → eval graph."""
    generation = dataclasses.replace(generation_step(version), override_path=_temporary_path(version, "tasks"))
    dataset = dataclasses.replace(
        dataset_step(generation, version=version, accepted_examples=ACCEPTED_EXAMPLES),
        override_path=_temporary_path(version, "chat"),
    )
    staged_model = _staged_model()
    store = dataclasses.replace(
        store_step(dataset, staged_model, version=version),
        override_path=_temporary_path(version, "store"),
    )
    sft = _sft_step(store, staged_model, version=version)
    evaluation_model = dataclasses.replace(
        _base_model(),
        name="snowball-curriculum-sft-finance",
        location="artifact://pending",
        tokenizer=SNOWBALL_TOKENIZER,
    )
    evaluation = eval_checkpoint_step(
        sft,
        evaluation_model,
        evalchemy_config_path=FINANCEBENCH_CONFIG,
        version=version,
        submission_cluster=COREWEAVE_CLUSTER,
        federated_cluster=COREWEAVE_CLUSTER,
    )
    return {"tasks": generation, "data": dataset, "store": store, "sft": sft, "eval": evaluation}


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(("tasks", "data", "store", "sft", "eval", "all")), default="all")
@build_options
def main(stage: str) -> dict[str, ArtifactStep]:
    version = resolve_version("curriculum-sft/snowball", None)
    pipeline = build_pipeline(version)
    return {stage: pipeline[stage]} if stage != "all" else {"eval": pipeline["eval"]}


if __name__ == "__main__":
    main()
