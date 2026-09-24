# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render verified finance tasks and build Snowball's Datakit token store."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pyarrow.parquet as pq
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_to_parquet
from marin.datakit.chat_render import render_chat_to_parquet
from marin.datakit.download.rollout_transforms import openai_chat_document
from marin.datakit.normalize import NormalizedData, normalize_to_parquet
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.processing.tokenize.attributes import TokenizeAttributesConfig, tokenize_attributes
from marin.processing.tokenize.store_builder import BuildLevanterStoreConfig, build_levanter_store
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.writers import write_parquet_file

from experiments.post_training.curriculum_sft.generated_tasks import GENERATION_FILENAME, task_payload
from experiments.post_training.curriculum_sft.task_rows import generated_payloads_to_rows

RAW_CHAT_FILENAME = "chat/part-00000-of-00001.parquet"
NORMALIZED_MAIN_RELATIVE_PATH = "normalized/outputs/main"
STORE_RELATIVE_PATH = "store"


class FinanceDataset(Artifact):
    """Datakit-rendered finance conversations."""

    main_output_dir: str


class FinanceStore(Artifact):
    """Token store consumed by Snowball SFT."""

    cache_path: str
    total_tokens: int


@dataclass(frozen=True)
class MaterializeDatasetConfig:
    generation_root: str
    output_path: str
    accepted_examples: int


@dataclass(frozen=True)
class BuildStoreConfig:
    normalized_path: str
    output_path: str
    tokenizer: str


def materialize_dataset(config: MaterializeDatasetConfig) -> FinanceDataset:
    """Select verified tasks and render canonical conversations to Parquet."""
    ledger = json.loads((StoragePath(config.generation_root) / GENERATION_FILENAME).read_text())
    task_data_path = prefix_join(config.generation_root, ledger["task_data"])
    with StoragePath(task_data_path).open("rb") as handle:
        records = pq.ParquetFile(handle).read().to_pylist()
    if len(records) != ledger["requested"]:
        raise ValueError(f"generation manifest records {ledger['requested']} tasks; Parquet contains {len(records)}")
    rows = generated_payloads_to_rows([task_payload(record) for record in records], config.accepted_examples)

    output = StoragePath(config.output_path)
    output.mkdirs()
    chat_path = output / RAW_CHAT_FILENAME
    chat_documents = [openai_chat_document(row["messages"], "curriculum-sft", source_id=row["id"]) for row in rows]
    write_parquet_file(chat_documents, str(chat_path), schema=CHAT_SCHEMA)

    normalized_chat = normalize_chat_to_parquet(
        input_path=str(chat_path.parent),
        output_path=str(output / "normalized-chat"),
        file_extensions=(".parquet",),
        max_workers=1,
    )
    rendered_path = output / "rendered"
    render_chat_to_parquet(
        input_path=normalized_chat.main_output_dir,
        output_path=str(rendered_path),
        max_workers=1,
    )
    normalized = normalize_to_parquet(
        input_path=str(rendered_path),
        output_path=str(output / "normalized"),
        file_extensions=(".parquet",),
        max_workers=1,
        bare=True,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {"generation_root": config.generation_root, "accepted_examples": len(rows)},
            indent=2,
        )
        + "\n"
    )
    return FinanceDataset(path=config.output_path, main_output_dir=normalized.main_output_dir)


def build_store(config: BuildStoreConfig) -> FinanceStore:
    """Tokenize rendered Parquet and build the packed Levanter store."""
    normalized = NormalizedData(
        main_output_dir=config.normalized_path,
        dup_output_dir=prefix_join(config.output_path, "unused-dups"),
        counters={},
    )
    tokenized = tokenize_attributes(
        TokenizeAttributesConfig(
            train_source=normalized,
            output_path=prefix_join(config.output_path, "tokenized"),
            tokenizer=config.tokenizer,
            tokenizer_backend=TokenizerBackend.HF,
            format=TextLmDatasetFormat(),
            max_workers=1,
        )
    )
    store = build_levanter_store(
        BuildLevanterStoreConfig(
            sources=[tokenized],
            cache_path=prefix_join(config.output_path, STORE_RELATIVE_PATH),
            max_workers=1,
        )
    )
    train = store.splits.get("train")
    if train is None or train.total_tokens <= 0:
        raise ValueError("curriculum SFT Datakit store produced no training tokens")
    return FinanceStore(path=config.output_path, cache_path=store.cache_path, total_tokens=train.total_tokens)


def dataset_step(
    generation: ArtifactStep[Artifact], *, version: str, accepted_examples: int
) -> ArtifactStep[FinanceDataset]:
    def build_config(ctx: StepContext) -> MaterializeDatasetConfig:
        return MaterializeDatasetConfig(
            generation_root=ctx.artifact_path(generation),
            output_path=ctx.output_path,
            accepted_examples=accepted_examples,
        )

    return ArtifactStep(
        name=user_owned_name("documents/curriculum-sft/finance-chat"),
        version=version,
        artifact_type=FinanceDataset,
        run=materialize_dataset,
        build_config=build_config,
        deps=(generation,),
    )


def store_step(
    dataset: ArtifactStep[FinanceDataset],
    tokenizer: ArtifactStep[Artifact],
    *,
    version: str,
) -> ArtifactStep[FinanceStore]:
    def build_config(ctx: StepContext) -> BuildStoreConfig:
        return BuildStoreConfig(
            normalized_path=prefix_join(ctx.artifact_path(dataset), NORMALIZED_MAIN_RELATIVE_PATH),
            output_path=ctx.output_path,
            tokenizer=ctx.artifact_path(tokenizer),
        )

    return ArtifactStep(
        name=user_owned_name("tokenized/curriculum-sft/finance"),
        version=version,
        artifact_type=FinanceStore,
        run=build_store,
        build_config=build_config,
        deps=(dataset, tokenizer),
    )
