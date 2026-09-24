# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.dataset as ds
from zephyr.writers import write_parquet_file

from experiments.post_training.curriculum_sft.data import MaterializeDatasetConfig, materialize_dataset
from experiments.post_training.curriculum_sft.generated_tasks import (
    GENERATED_TASK_SCHEMA,
    GENERATED_TASKS_FILENAME,
    GENERATION_FILENAME,
    generated_task_record,
    task_payload,
)


def test_generated_task_parquet_preserves_rejected_payloads(tmp_path):
    payloads = [
        {
            "task_id": "bad-arithmetic",
            "issuer": "Fictional issuer",
            "question": "Revenue is 100 and operating cost is 80. What is gross profit?",
            "facts": {"revenue": 100, "operating_cost": 80},
            "answer": {"gross_profit": 21, "margin_bps": 2000},
            "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
        },
        {
            "issuer": "Fictional issuer",
            "question": "Revenue is 100 and operating cost is 80. What is gross profit?",
            "facts": {"revenue": 100, "operating_cost": 80},
            "answer": {"gross_profit": 20, "margin_bps": 2000},
            "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
        },
    ]
    parquet_path = tmp_path / "tasks.parquet"
    write_parquet_file(
        [
            generated_task_record(batch_index=index, seed=19 + index, payload=payload)
            for index, payload in enumerate(payloads)
        ],
        str(parquet_path),
        schema=GENERATED_TASK_SCHEMA,
    )

    records = ds.dataset(parquet_path, format="parquet").to_table().to_pylist()

    assert [record["batch_index"] for record in records] == [0, 1]
    assert [record["seed"] for record in records] == [19, 20]
    assert [record["accepted"] for record in records] == [False, False]
    assert [task_payload(record) for record in records] == payloads


def test_materialize_dataset_produces_rendered_parquet(tmp_path):
    payload = {
        "task_id": "example",
        "issuer": "Fictional issuer",
        "question": "Revenue is 80 and operating cost is 100. What is gross profit?",
        "facts": {"revenue": 100, "operating_cost": 80},
        "answer": {"gross_profit": 20, "margin_bps": 2000},
        "evidence": ["disclosure.revenue", "disclosure.operating_cost"],
    }
    ledger = {
        "batch_id": "batch-test",
        "task_data": GENERATED_TASKS_FILENAME,
        "requested": 1,
        "accepted": 1,
        "unique_accepted": 1,
    }
    generation_root = tmp_path / "generation"
    generation_root.mkdir()
    (generation_root / GENERATION_FILENAME).write_text(json.dumps(ledger))
    task_path = generation_root / GENERATED_TASKS_FILENAME
    task_path.parent.mkdir()
    write_parquet_file(
        [generated_task_record(batch_index=0, seed=17, payload=payload)],
        str(task_path),
        schema=GENERATED_TASK_SCHEMA,
    )

    result = materialize_dataset(
        MaterializeDatasetConfig(
            generation_root=str(generation_root),
            output_path=str(tmp_path / "output"),
            accepted_examples=1,
        )
    )

    table = ds.dataset(result.main_output_dir, format="parquet").to_table()
    assert table.column_names == ["id", "text", "source_id"]
    assert table.num_rows == 1
    [text] = table["text"].to_pylist()
    assert "revenue of 100" in text
    assert "operating cost of 80" in text
    assert "Revenue is 80" not in text
