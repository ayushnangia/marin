# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate structured finance tasks from one pinned curriculum capability."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from pydantic import Field
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_parquet_file

from experiments.post_training.curriculum_sft.generated_tasks import (
    GENERATED_TASK_SCHEMA,
    GENERATED_TASKS_FILENAME,
    GENERATION_FILENAME,
    RAW_RESPONSES_FILENAME,
    generated_task_record,
    task_payload,
)
from experiments.post_training.curriculum_sft.task_rows import unique_accepted_payloads
from experiments.post_training.curriculum_sft.tasks import BASIS_POINTS_SCALE, EVIDENCE_IDS
from experiments.post_training.glm import (
    DEFAULT_GLM_RELAY_JOB,
    GLM_BULK_TOKEN_ENV,
    GLM_MODEL,
    resolve_glm_base_url,
)
from experiments.post_training.task_curriculum.models import StrictModel

logger = logging.getLogger(__name__)

SUBJECT_AREA = "financial reporting"
TASK_FAMILY = "fictional company profitability questions"
CURRICULUM_PACKET = "\n".join(
    (
        "Curriculum catalog: 2026.09.20-cross-domain-v3",
        "Subject curriculum: D27 production-candidate-v1",
        "Capability ID: d27.reporting.analysis",
        "Name: Calculate Reported Financial Measures",
        "Outcome: Select, calculate, and reconcile comparable financial measures "
        "from statements and disclosures under stated definitions.",
        "Includes:",
        "- profitability, liquidity, leverage, and efficiency ratios",
        "- common-size analysis",
        "- trend and growth measures",
        "- earnings-quality and cash-versus-profit measures",
        "Excludes:",
        "- standalone disclosure lookup",
        "- forward-looking enterprise valuation",
        "- evidence-attributed driver bridges",
    )
)
ACCEPTED_EXAMPLES = 16
TASKS_PER_BATCH = 16
BATCH_COUNT = 4
SEED = 17


class FinanceFacts(StrictModel):
    revenue: int
    operating_cost: int


class FinanceAnswer(StrictModel):
    gross_profit: int
    margin_bps: int


class GeneratedFinanceTask(StrictModel):
    task_id: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    facts: FinanceFacts
    question: str = Field(min_length=1)
    answer: FinanceAnswer
    evidence: list[str]


class GeneratedFinanceTasks(StrictModel):
    tasks: list[GeneratedFinanceTask]


GENERATED_TASKS_TOOL = StructuredTool(
    name="submit_tasks",
    description="Submit the generated fictional tasks.",
    output_type=GeneratedFinanceTasks,
)


@dataclass(frozen=True)
class GenerateTasksConfig:
    output_path: str
    curriculum_packet: str
    subject_area: str
    task_family: str
    accepted_examples: int
    tasks_per_batch: int
    batch_count: int
    seed: int
    relay_job: str


def generation_prompt(config: GenerateTasksConfig) -> str:
    """Specify the executable finance task without benchmark examples."""
    evidence_ids = " and ".join(EVIDENCE_IDS)
    return (
        f"Subject area: {config.subject_area}\nTask family: {config.task_family}\n"
        f"Curriculum guidance:\n{config.curriculum_packet}\n"
        f"Generate {config.tasks_per_batch} distinct fictional tasks through the provided tool. "
        "Do not use or imitate benchmark examples. Every question must ask for gross profit "
        "(revenue minus operating cost) and gross margin in basis points "
        f"(gross profit divided by revenue times {BASIS_POINTS_SCALE}). "
        "Use positive integer revenue and operating_cost with operating_cost below revenue. "
        "Choose values whose basis-point answer is integral, make each answer consistent with its facts, "
        f"and cite exactly the evidence IDs {evidence_ids}. Vary fictional issuers, values, and wording."
    )


def _request(config: GenerateTasksConfig, batch_index: int) -> dict[str, Any]:
    body = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": "Generate fictional tasks and call submit_tasks exactly once."},
            {"role": "user", "content": generation_prompt(config)},
        ],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0.8,
        "seed": config.seed + batch_index,
        "max_tokens": 12000,
    }
    body.update(GENERATED_TASKS_TOOL.request_fields())
    return {
        "custom_id": f"finance-{batch_index}",
        "method": "POST",
        "url": CHAT_COMPLETIONS_ENDPOINT,
        "body": body,
    }


def _parse_output(raw_output: str, config: GenerateTasksConfig) -> list[dict[str, Any]]:
    batches: dict[str, list[dict[str, Any]]] = {}
    expected_ids = {f"finance-{index}" for index in range(config.batch_count)}
    for line in raw_output.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        custom_id = response["custom_id"]
        if custom_id not in expected_ids or custom_id in batches:
            raise ValueError(f"unexpected or duplicate GLM request ID: {custom_id}")
        result = response.get("response") or {}
        if response.get("error") or result.get("status_code") != 200:
            raise RuntimeError(f"GLM request {custom_id} failed")
        tasks = [task.model_dump(mode="json") for task in GENERATED_TASKS_TOOL.parse(result["body"]).tasks]
        if len(tasks) != config.tasks_per_batch:
            raise ValueError(f"GLM request {custom_id} returned {len(tasks)} tasks, expected {config.tasks_per_batch}")
        batches[custom_id] = tasks
    if set(batches) != expected_ids:
        raise RuntimeError(f"GLM batch omitted requests: {sorted(expected_ids - set(batches))}")
    return [
        generated_task_record(batch_index=index, seed=config.seed + index, payload=task)
        for index in range(config.batch_count)
        for task in batches[f"finance-{index}"]
    ]


def generate_tasks(config: GenerateTasksConfig) -> Artifact:
    """Write generated tasks, oracle outcomes, and exact provider responses."""
    client = OpenAIBatchClient(resolve_glm_base_url(config.relay_job), os.environ[GLM_BULK_TOKEN_ENV])
    requests = [_request(config, index) for index in range(config.batch_count)]
    submission = client.submit(requests, "curriculum-finance-tasks.jsonl")
    batch_output = client.output(client.wait(submission.batch_id, 5.0))
    if batch_output.errors:
        raise RuntimeError(f"GLM batch {submission.batch_id} returned errors")
    records = _parse_output(batch_output.output, config)
    accepted = [record for record in records if record["accepted"]]
    unique_accepted = unique_accepted_payloads([task_payload(record) for record in accepted])
    if len(unique_accepted) < config.accepted_examples:
        raise ValueError(f"only {len(unique_accepted)} unique verified tasks; need {config.accepted_examples}")
    output = StoragePath(config.output_path)
    output.mkdirs()
    tasks_path = output / GENERATED_TASKS_FILENAME
    tasks_path.parent.mkdirs()
    write_parquet_file(records, str(tasks_path), schema=GENERATED_TASK_SCHEMA)
    (output / RAW_RESPONSES_FILENAME).write_text(batch_output.output.rstrip() + "\n")
    (output / GENERATION_FILENAME).write_text(
        json.dumps(
            {
                "batch_id": submission.batch_id,
                "curriculum_packet": config.curriculum_packet,
                "task_data": GENERATED_TASKS_FILENAME,
                "raw_responses": RAW_RESPONSES_FILENAME,
                "requested": len(records),
                "accepted": sum(record["accepted"] for record in records),
                "unique_accepted": len(unique_accepted),
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("generated %s verified tasks from %s attempts", len(unique_accepted), len(records))
    return Artifact(path=config.output_path)


def generation_step(version: str) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> GenerateTasksConfig:
        return GenerateTasksConfig(
            output_path=ctx.output_path,
            curriculum_packet=CURRICULUM_PACKET,
            subject_area=SUBJECT_AREA,
            task_family=TASK_FAMILY,
            accepted_examples=ACCEPTED_EXAMPLES,
            tasks_per_batch=TASKS_PER_BATCH,
            batch_count=BATCH_COUNT,
            seed=SEED,
            relay_job=DEFAULT_GLM_RELAY_JOB,
        )

    return ArtifactStep(
        name=user_owned_name("documents/curriculum-sft/finance-tasks"),
        version=version,
        artifact_type=Artifact,
        run=generate_tasks,
        build_config=build_config,
    )
