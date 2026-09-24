# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn verified finance tasks into Snowball conversations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.post_training.curriculum_sft.tasks import BASIS_POINTS_SCALE, SyntheticFinanceTask, task_from_payload
from experiments.post_training.curriculum_sft.verification import verify_task_payload

SFT_SYSTEM_PROMPT = (
    "Solve the fictional financial calculation. Return exactly one JSON object with keys result and evidence. "
    "result must contain integer gross_profit and margin_bps; evidence must contain the cited evidence IDs."
)


def _conversation(task: SyntheticFinanceTask) -> dict[str, Any]:
    answer = json.dumps(
        {"result": task.expected_result, "evidence": list(task.evidence_ids)},
        separators=(",", ":"),
    )
    return {
        "id": task.task_id,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"A fictional issuer reports revenue of {task.revenue} "
                    f"(evidence: {task.evidence_ids[0]}) and operating cost of {task.operating_cost} "
                    f"(evidence: {task.evidence_ids[1]}). Calculate gross profit as revenue minus operating cost "
                    f"and gross margin in basis points as gross profit divided by revenue times {BASIS_POINTS_SCALE}."
                ),
            },
            {"role": "assistant", "content": answer},
        ],
    }


def unique_accepted_payloads(payloads: Sequence[object]) -> list[Mapping[str, Any]]:
    """Keep oracle-accepted tasks with distinct IDs, questions, and fact pairs."""
    accepted: list[Mapping[str, Any]] = []
    task_ids: set[str] = set()
    questions: set[str] = set()
    fact_tuples: set[tuple[int, int]] = set()
    for payload in payloads:
        if not isinstance(payload, Mapping) or not verify_task_payload(payload).accepted:
            continue
        facts = payload["facts"]
        task_id = payload["task_id"]
        question = " ".join(payload["question"].lower().split())
        fact_tuple = (facts["revenue"], facts["operating_cost"])
        if task_id in task_ids or question in questions or fact_tuple in fact_tuples:
            continue
        task_ids.add(task_id)
        questions.add(question)
        fact_tuples.add(fact_tuple)
        accepted.append(payload)
    return accepted


def generated_payloads_to_rows(payloads: Sequence[object], accepted_examples: int) -> list[dict[str, Any]]:
    """Use verified facts to build consistent questions and answers."""
    accepted = unique_accepted_payloads(payloads)[:accepted_examples]
    if len(accepted) != accepted_examples:
        raise ValueError(f"found {len(accepted)} unique oracle-accepted tasks; needed {accepted_examples}")
    return [_conversation(task_from_payload(dict(payload))) for payload in accepted]
