from __future__ import annotations

from typing import Any


class AdaptiveExecutionFabric:
    """
    LeanTrader execution-environment registry.

    LeanTrader intelligence is independent of paper, Testnet/demo, or live.
    Environments are attached beneath the trading core.

    This class does not create trading signals and does not grant live authority.
    """

    VERSION = "1.62.0"

    def __init__(self) -> None:
        self.environments: dict[str, dict[str, Any]] = {}
        self.market_capabilities: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def register_environment(
        self,
        name: str,
        *,
        role: str,
        adapter: Any | None,
        market_types: tuple[str, ...],
        directions: tuple[str, ...],
        externally_executable: bool,
    ) -> None:
        normalized = str(name).strip().lower()

        if not normalized:
            raise ValueError("execution environment name is required")

        if role not in {"training", "validation", "live"}:
            raise ValueError("invalid execution environment role")

        self.environments[normalized] = {
            "name": normalized,
            "role": role,
            "adapter": adapter,
            "market_types": tuple(
                dict.fromkeys(
                    str(value).strip().lower()
                    for value in market_types
                    if str(value).strip()
                )
            ),
            "directions": tuple(
                dict.fromkeys(
                    str(value).strip().lower()
                    for value in directions
                    if str(value).strip()
                )
            ),
            "externally_executable": bool(
                externally_executable
            ),
        }

    def declare_market_capability(
        self,
        market_type: str,
        *,
        directions: tuple[str, ...],
        data_sources: tuple[str, ...],
        executable_environments: tuple[str, ...] = (),
    ) -> None:
        key = str(market_type).strip().lower()

        self.market_capabilities[key] = {
            "market_type": key,
            "directions": list(
                dict.fromkeys(directions)
            ),
            "data_sources": list(
                dict.fromkeys(data_sources)
            ),
            "real_data": True,
            "executable_environments": list(
                dict.fromkeys(
                    str(value).strip().lower()
                    for value in executable_environments
                    if str(value).strip()
                )
            ),
            "execution_authority": False,
        }

    def execute_events(
        self,
        environment: str,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        name = str(environment).strip().lower()
        row = self.environments.get(name)

        if not row:
            raise RuntimeError(
                f"execution environment not registered: {name}"
            )

        if row["role"] == "live":
            raise RuntimeError(
                "live execution adapter is not active in this validation runtime"
            )

        if not row["externally_executable"]:
            raise RuntimeError(
                f"{name} is not an external execution sink"
            )

        adapter = row.get("adapter")

        if adapter is None:
            raise RuntimeError(
                f"{name} has no execution adapter"
            )

        method = getattr(
            adapter,
            "mirror_events",
            None,
        )

        if not callable(method):
            raise RuntimeError(
                f"{name} adapter cannot execute event batches"
            )

        return list(
            method(events)
            or []
        )

    def environment_health(
        self,
        environment: str,
    ) -> dict[str, Any]:
        name = str(environment).strip().lower()
        row = self.environments.get(name)

        if not row:
            return {
                "available": False,
                "environment": name,
            }

        adapter = row.get("adapter")
        adapter_health: dict[str, Any] = {}

        if adapter is not None:
            health = getattr(
                adapter,
                "health",
                None,
            )

            if callable(health):
                try:
                    result = health()
                    if isinstance(result, dict):
                        adapter_health = result
                except Exception as exc:
                    adapter_health = {
                        "healthy": False,
                        "error": type(exc).__name__,
                    }

        return {
            "available": True,
            "environment": name,
            "role": row["role"],
            "market_types": list(
                row["market_types"]
            ),
            "directions": list(
                row["directions"]
            ),
            "externally_executable": row[
                "externally_executable"
            ],
            "adapter_health": adapter_health,
        }

    def health(self) -> dict[str, Any]:
        environments = {
            name: self.environment_health(name)
            for name in sorted(
                self.environments
            )
        }

        return {
            "version": self.VERSION,
            "system_identity": "leantrader",
            "execution_environment_is_not_system_identity": True,
            "environments": environments,
            "market_capabilities": self.market_capabilities,
            "paper_role": "internal_training_and_simulation",
            "testnet_role": "authenticated_execution_validation",
            "live_role": "explicit_operator_selected_execution",
            "live_adapter_active": any(
                row.get("role") == "live"
                and row.get(
                    "externally_executable"
                )
                for row in self.environments.values()
            ),
            "live_authority": False,
        }
