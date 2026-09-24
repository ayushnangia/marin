# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.model_config import ModelConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint

from experiments.evaluation.pipeline import eval_checkpoint_step


def test_eval_checkpoint_step_depends_on_export_and_drops_source_revision(tmp_path):
    checkpoint = ArtifactStep(
        name="checkpoints/generated-model",
        version="2026.09.22",
        artifact_type=LevanterCheckpoint,
        run=lambda _config: None,
        build_config=lambda _ctx: {},
    )
    source = ModelConfig(
        name="generated-model",
        location="organization/base-model",
        revision="base-model-revision",
        tokenizer="organization/tokenizer",
    )
    config_path = tmp_path / "financebench.yaml"
    config_path.write_text("tasks: []\n")
    step = eval_checkpoint_step(checkpoint, source, evalchemy_config_path=config_path, version="2026.09.22")

    config = step.build_config(StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps))

    assert step.deps == (checkpoint,)
    assert config.model.location == "artifact://checkpoints/generated-model@2026.09.22"
    assert config.model.revision is None
