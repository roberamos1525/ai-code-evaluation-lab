"""Wiring of settings into the long-lived service objects shared by the CLI and the API."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from codeeval.config import Settings
from codeeval.datasets.registry import DatasetRegistry
from codeeval.engine.config import RunConfig
from codeeval.engine.runner import EvaluationEngine
from codeeval.sandbox import create_backend
from codeeval.sandbox.base import SandboxBackend
from codeeval.sandbox.policy import PolicyConfig
from codeeval.tracking.store import ExperimentStore


@dataclass
class AppContext:
    settings: Settings
    store: ExperimentStore
    registry: DatasetRegistry
    backend: SandboxBackend
    engine: EvaluationEngine

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        backend: SandboxBackend | None = None,
        http_client: httpx.Client | None = None,
    ) -> AppContext:
        store = ExperimentStore(settings.database_path)
        registry = DatasetRegistry(settings.dataset_dirs)
        backend = backend or create_backend(
            settings.sandbox_backend,
            python_executable=settings.python_executable,
            root_dir=settings.sandbox_root,
            docker_config=settings.docker_config(),
            keep_workdirs=settings.keep_workdirs,
        )
        engine = EvaluationEngine(
            store,
            registry,
            backend,
            allow_network=settings.allow_network_adapters,
            http_client=http_client,
            max_samples_per_run=settings.max_samples_per_run,
        )
        return cls(settings, store, registry, backend, engine)

    def default_run_config(self, **overrides: object) -> RunConfig:
        """A RunConfig seeded with the settings' default limits, policy and workers."""
        values: dict[str, object] = {
            "limits": self.settings.default_limits(),
            "policy": PolicyConfig(strictness=self.settings.strictness),
            "workers": self.settings.workers,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return RunConfig(**values)  # type: ignore[arg-type]

    def close(self) -> None:
        self.store.close()
