import asyncio
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import threading
import time


ENGINE_NAME = re.compile(
    r"(engine|brain|swarm|scanner|trader|"
    r"orchestrator|intelligence|harmony|"
    r"apex|quantum|scalp|arbitrage|"
    r"evolution|learner|risk|core)",
    re.I,
)


PROTECTED_TYPES = {
    (
        "EVOLUTION_ENGINE",
        "ULTIMATE_EVOLUTION_ENGINE",
    ),
    (
        "ultra_arbitrage_engine",
        "UltraArbitrageEngine",
    ),
    (
        "ultra_scalping_engine",
        "UltraScalpingEngine",
    ),
    (
        "working_450_models_bot",
        "UltimateBot450Models",
    ),
    (
        "ultra_swarm_consciousness",
        "SwarmConsciousness",
    ),
    (
        "ultra_backtest_engine",
        "UltraBacktestEngine",
    ),
}


def _engine_object(value):
    if value is None:
        return False

    if isinstance(
        value,
        (
            str,
            bytes,
            int,
            float,
            bool,
            list,
            tuple,
            set,
            dict,
        ),
    ):
        return False

    if (
        inspect.ismodule(value)
        or inspect.isclass(value)
        or inspect.isfunction(value)
        or inspect.ismethod(value)
    ):
        return False

    return True


def collect_engine_heads(
    orchestrator,
):
    objects = {}

    def add(
        value,
        alias,
    ):
        if not _engine_object(
            value
        ):
            return

        identity = id(value)

        row = objects.setdefault(
            identity,
            {
                "object": value,
                "aliases": [],
            },
        )

        if alias not in row[
            "aliases"
        ]:
            row[
                "aliases"
            ].append(
                alias
            )

    for mapping_name in (
        "trading_engines",
        "ai_systems",
        "advanced_systems",
        "advanced_orchestrators",
        "orchestrators",
    ):
        mapping = getattr(
            orchestrator,
            mapping_name,
            None,
        )

        if not isinstance(
            mapping,
            dict,
        ):
            continue

        for name, value in (
            mapping.items()
        ):
            add(
                value,
                f"{mapping_name}.{name}",
            )

    for name, value in vars(
        orchestrator
    ).items():

        if not ENGINE_NAME.search(
            name
        ):
            continue

        add(
            value,
            f"attr.{name}",
        )

    output = []

    for identity, row in (
        objects.items()
    ):
        value = row[
            "object"
        ]

        cls = value.__class__

        start_methods = []

        for name in (
            "start",
            "run",
            "start_engine",
            "start_evolution",
            "start_evolution_cycle",
            "start_arbitrage_scanning",
            "start_scalping",
            "start_swarm_consciousness",
            "start_testnet_trading",
            "run_ultimate_bot",
        ):
            if callable(
                getattr(
                    value,
                    name,
                    None,
                )
            ):
                start_methods.append(
                    name
                )

        status = getattr(
            value,
            "status",
            None,
        )

        output.append(
            {
                "object_id":
                    identity,
                "module":
                    cls.__module__,
                "class":
                    cls.__name__,
                "aliases":
                    sorted(
                        row[
                            "aliases"
                        ]
                    ),
                "status":
                    (
                        str(status)
                        if status
                        is not None
                        else "ACTIVE"
                    ),
                "start_methods":
                    start_methods,
            }
        )

    output.sort(
        key=lambda item: (
            item["module"],
            item["class"],
            item["aliases"],
        )
    )

    return output


def protected_duplicates(
    heads,
):
    grouped = {}

    for row in heads:
        key = (
            row["module"],
            row["class"],
        )

        grouped.setdefault(
            key,
            []
        ).append(
            row
        )

    output = []

    for key in PROTECTED_TYPES:
        rows = grouped.get(
            key,
            [],
        )

        if len(rows) > 1:
            output.append(
                {
                    "module":
                        key[0],
                    "class":
                        key[1],
                    "count":
                        len(rows),
                    "aliases": [
                        row["aliases"]
                        for row in rows
                    ],
                }
            )

    return output


def _constructor_kwargs(
    cls,
    orchestrator,
):
    signature = inspect.signature(
        cls
    )

    universe = (
        getattr(
            orchestrator,
            "trading_universe",
            None,
        )
        or getattr(
            getattr(
                orchestrator,
                "ultra_core",
                None,
            ),
            "universe",
            None,
        )
    )

    known = {
        "ultra_core":
            getattr(
                orchestrator,
                "ultra_core",
                None,
            ),
        "risk_engine":
            getattr(
                orchestrator,
                "risk_engine",
                None,
            ),
        "data_hub":
            getattr(
                orchestrator,
                "data_hub",
                None,
            ),
        "router":
            getattr(
                orchestrator,
                "router",
                None,
            ),
        "universe":
            universe,
        "logger":
            logging.getLogger(
                "apex_predator"
            ),
        "exchanges":
            getattr(
                orchestrator,
                "exchanges",
                None,
            ),
    }

    kwargs = {}

    missing = []

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in {
            "self",
            "args",
            "kwargs",
        }:
            continue

        if name in known:
            kwargs[
                name
            ] = known[
                name
            ]

        elif (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ):
            missing.append(
                name
            )

    if missing:
        raise RuntimeError(
            "Apex required constructor "
            "parameters are not mapped: "
            + ",".join(missing)
        )

    return kwargs


def restore_apex(
    orchestrator,
    logger,
):
    module = importlib.import_module(
        "ultra_rare.goldmine."
        "APEX_PREDATOR_INTELLIGENCE"
    )

    apex_cls = getattr(
        module,
        "ApexPredatorIntelligence",
    )

    kwargs = _constructor_kwargs(
        apex_cls,
        orchestrator,
    )

    apex = apex_cls(
        **kwargs
    )

    specialists = []

    for name, value in vars(
        module
    ).items():
        if (
            inspect.isclass(value)
            and value.__module__
            == module.__name__
            and name
            != "ApexPredatorIntelligence"
        ):
            specialists.append(
                name
            )

    if len(
        specialists
    ) != 7:
        raise RuntimeError(
            "Expected 7 historical Apex "
            f"specialists; found {len(specialists)}"
        )

    setattr(
        orchestrator,
        "apex_predator",
        apex,
    )

    advanced = getattr(
        orchestrator,
        "advanced_systems",
        None,
    )

    if isinstance(
        advanced,
        dict,
    ):
        advanced[
            "apex_predator"
        ] = apex

    logger.info(
        "🧬 APEX PREDATOR RESTORED "
        "| specialist_classes=%s "
        "| execution_influence="
        "EVIDENCE_AUDIT_PENDING",
        len(specialists),
    )

    return (
        apex,
        specialists,
    )


def _register_harmony_head(
    harmonizer,
    row,
):
    method = (
        harmonizer.register_engine
    )

    signature = inspect.signature(
        method
    )

    name = (
        row["aliases"][0]
        if row["aliases"]
        else (
            row["module"]
            + "."
            + row["class"]
        )
    )

    values = {
        "name": name,
        "engine_name": name,
        "engine_type":
            row["class"],
        "type":
            row["class"],
        "base_frequency":
            None,
        "frequency":
            None,
    }

    args = []
    kwargs = []

    for parameter in (
        signature.parameters.values()
    ):
        if parameter.name == "self":
            continue

        if parameter.name in values:
            value = values[
                parameter.name
            ]

            if (
                value is None
                and parameter.default
                is not inspect.Parameter.empty
            ):
                continue

            args.append(
                value
            )

        elif (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ):
            raise RuntimeError(
                "Unsupported ConsciousnessHarmonizer "
                "register parameter: "
                + parameter.name
            )

    method(
        *args
    )



# PASS4_REAL_RUNTIME_PULSE_BRIDGE
def _runtime_engine_objects(
    orchestrator,
):
    """
    Return one canonical object per identity.

    Aliases do not create additional heads.
    """
    objects = {}

    def add(
        value,
        alias,
    ):
        if not _engine_object(
            value
        ):
            return

        identity = id(value)

        row = objects.setdefault(
            identity,
            {
                "object":
                    value,
                "aliases":
                    [],
            },
        )

        if alias not in row[
            "aliases"
        ]:
            row[
                "aliases"
            ].append(
                alias
            )

    for mapping_name in (
        "trading_engines",
        "ai_systems",
        "advanced_systems",
        "advanced_orchestrators",
        "orchestrators",
    ):
        mapping = getattr(
            orchestrator,
            mapping_name,
            None,
        )

        if not isinstance(
            mapping,
            dict,
        ):
            continue

        for name, value in (
            mapping.items()
        ):
            add(
                value,
                f"{mapping_name}.{name}",
            )

    for name, value in vars(
        orchestrator
    ).items():

        if not ENGINE_NAME.search(
            name
        ):
            continue

        add(
            value,
            f"attr.{name}",
        )

    output = []

    for identity, row in (
        objects.items()
    ):
        aliases = sorted(
            row["aliases"]
        )

        output.append({
            "identity":
                identity,
            "object":
                row["object"],
            "alias":
                (
                    aliases[0]
                    if aliases
                    else str(identity)
                ),
            "aliases":
                aliases,
        })

    return output


def _scalar_activity_snapshot(
    obj,
):
    snapshot = {}

    tokens = (
        "cycle",
        "count",
        "counter",
        "signal",
        "trade",
        "fill",
        "scan",
        "update",
        "heartbeat",
        "iteration",
        "generation",
        "step",
    )

    try:
        values = vars(obj)
    except Exception:
        return snapshot

    for name, value in (
        values.items()
    ):
        lower = name.lower()

        if not any(
            token in lower
            for token in tokens
        ):
            continue

        if isinstance(
            value,
            (
                bool,
                int,
                float,
                str,
            ),
        ):
            snapshot[
                name
            ] = value

    return snapshot


def _runtime_activity_evidence(
    obj,
    previous_snapshot,
):
    """
    Return objective liveness evidence only.

    This is NOT trading confidence and is never
    used as financial evidence.
    """
    evidence = []

    # Explicit runtime flags.
    for name in (
        "running",
        "is_running",
        "active",
        "started",
        "trading",
        "harmonizing",
        "_leantrader_arbitrage_started",
        "_leantrader_scalping_started",
        "_leantrader_swarm_started",
        "_leantrader_testnet_engine_started",
        "_leantrader_450_started",
        "_evolution_threads_started",
        "_active_engine_threads_started",
    ):
        value = getattr(
            obj,
            name,
            None,
        )

        if value is True:
            evidence.append(
                "flag:"
                + name
            )

    # Explicit status strings.
    status = getattr(
        obj,
        "status",
        None,
    )

    if isinstance(
        status,
        str,
    ):
        normalized = (
            status
            .strip()
            .lower()
        )

        if normalized in {
            "active",
            "running",
            "online",
            "trading",
            "started",
            "connected",
        }:
            evidence.append(
                "status:"
                + normalized
            )

    # Alive worker threads/tasks.
    try:
        values = vars(obj)
    except Exception:
        values = {}

    for name, value in (
        values.items()
    ):
        if isinstance(
            value,
            threading.Thread,
        ):
            if value.is_alive():
                evidence.append(
                    "thread:"
                    + name
                )

        elif isinstance(
            value,
            asyncio.Task,
        ):
            if not value.done():
                evidence.append(
                    "task:"
                    + name
                )

    # Recent real activity timestamps.
    now = time.time()

    for name, value in (
        values.items()
    ):
        lower = name.lower()

        if not (
            lower.startswith(
                "last_"
            )
            or lower.endswith(
                "_timestamp"
            )
        ):
            continue

        if not isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            continue

        # Unix timestamp range only.
        if value < 1_000_000_000:
            continue

        age = now - float(value)

        if 0 <= age <= 120:
            evidence.append(
                "recent:"
                + name
            )

    current_snapshot = (
        _scalar_activity_snapshot(
            obj
        )
    )

    if previous_snapshot is not None:
        for key, value in (
            current_snapshot.items()
        ):
            if (
                key
                in previous_snapshot
                and previous_snapshot[
                    key
                ] != value
            ):
                evidence.append(
                    "changed:"
                    + key
                )

    return (
        evidence,
        current_snapshot,
    )


def _write_harmony_runtime(
    harmonizer,
    active,
    evidence,
):
    health = (
        harmonizer
        .get_system_health()
    )

    payload = {
        "timestamp":
            time.time(),
        "health":
            health,
        "observed_active":
            sorted(
                active
            ),
        "activity_evidence":
            evidence,
    }

    root = Path(
        os.getenv(
            "LEANTRADER_DATA_DIR",
            "/app/data",
        )
    )

    target = (
        root
        / "runtime"
        / "harmony_runtime.json"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        target.with_suffix(
            ".tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        target
    )


def _start_runtime_pulse_bridge(
    orchestrator,
    harmonizer,
    logger,
):
    """
    Feed Harmony only from observed runtime
    liveness.

    No random values, no synthetic signals,
    no fake trading observations.
    """
    snapshots = {}

    def loop():
        while True:
            try:
                rows = (
                    _runtime_engine_objects(
                        orchestrator
                    )
                )

                active = []
                evidence_report = {}

                for row in rows:
                    obj = row[
                        "object"
                    ]

                    alias = row[
                        "alias"
                    ]

                    if alias not in (
                        harmonizer.engines
                    ):
                        continue

                    previous = (
                        snapshots.get(
                            row[
                                "identity"
                            ]
                        )
                    )

                    (
                        evidence,
                        snapshot,
                    ) = (
                        _runtime_activity_evidence(
                            obj,
                            previous,
                        )
                    )

                    snapshots[
                        row[
                            "identity"
                        ]
                    ] = snapshot

                    if not evidence:
                        continue

                    active.append(
                        alias
                    )

                    evidence_report[
                        alias
                    ] = evidence

                    # Energy here means measured
                    # runtime liveness only.
                    energy = min(
                        1.0,
                        0.20
                        + (
                            0.10
                            * len(
                                evidence
                            )
                        ),
                    )

                    harmonizer.pulse(
                        alias,
                        energy,
                    )

                _write_harmony_runtime(
                    harmonizer,
                    active,
                    evidence_report,
                )

            except Exception:
                logger.exception(
                    "Runtime Harmony pulse "
                    "bridge failed"
                )

            time.sleep(
                1.0
            )

    thread = threading.Thread(
        target=loop,
        name=(
            "leantrader-harmony-"
            "runtime-pulse"
        ),
        daemon=True,
    )

    thread.start()

    setattr(
        orchestrator,
        "_harmony_pulse_thread",
        thread,
    )

    logger.info(
        "💓 REAL RUNTIME HARMONY PULSE "
        "BRIDGE ONLINE"
    )

    return thread


def start_harmony(
    orchestrator,
    heads,
    logger,
):
    module = importlib.import_module(
        "ai.intelligence."
        "CONSCIOUSNESS_HARMONIZER"
    )

    harmony_cls = getattr(
        module,
        "ConsciousnessHarmonizer",
    )

    harmonizer = harmony_cls()

    registered = 0

    for row in heads:
        _register_harmony_head(
            harmonizer,
            row,
        )

        registered += 1

    setattr(
        orchestrator,
        "frequency_harmony",
        harmonizer,
    )

    setattr(
        orchestrator,
        "consciousness_harmonizer",
        harmonizer,
    )

    advanced = getattr(
        orchestrator,
        "advanced_systems",
        None,
    )

    if isinstance(
        advanced,
        dict,
    ):
        advanced[
            "frequency_harmony"
        ] = harmonizer

    starter = getattr(
        harmonizer,
        "start_harmonization",
        None,
    )

    if callable(starter):
        def run():
            try:
                result = starter()

                if inspect.isawaitable(
                    result
                ):
                    asyncio.run(
                        result
                    )

            except Exception:
                logger.exception(
                    "Frequency Harmony "
                    "background loop failed"
                )

        threading.Thread(
            target=run,
            name=(
                "leantrader-frequency-harmony"
            ),
            daemon=True,
        ).start()

    # PASS4_START_REAL_RUNTIME_PULSES
    _start_runtime_pulse_bridge(
        orchestrator,
        harmonizer,
        logger,
    )

    return (
        harmonizer,
        registered,
    )


def write_registry(
    orchestrator,
    apex_specialists,
    harmony_registered,
):
    heads = collect_engine_heads(
        orchestrator
    )

    duplicates = (
        protected_duplicates(
            heads
        )
    )

    swarm_agents = 0

    for row in heads:
        aliases = row.get(
            "aliases",
            []
        )

        for alias in aliases:
            value = None

            if alias.startswith(
                "attr."
            ):
                value = getattr(
                    orchestrator,
                    alias[5:],
                    None,
                )

            if value is not None:
                agents = getattr(
                    value,
                    "agents",
                    None,
                )

                if isinstance(
                    agents,
                    list,
                ):
                    swarm_agents = max(
                        swarm_agents,
                        len(agents),
                    )

    harmony = getattr(
        orchestrator,
        "frequency_harmony",
        None,
    )

    health = {}

    if (
        harmony is not None
        and callable(
            getattr(
                harmony,
                "get_system_health",
                None,
            )
        )
    ):
        try:
            health = (
                harmony
                .get_system_health()
                or {}
            )
        except Exception:
            health = {}

    report = {
        "timestamp":
            time.time(),
        "unique_heads":
            len(heads),
        "swarm_agents":
            swarm_agents,
        "protected_duplicates":
            duplicates,
        "apex_specialists":
            apex_specialists,
        "harmony_registered_heads":
            harmony_registered,
        "harmony_health":
            health,
        "heads":
            heads,
    }

    root = Path(
        os.getenv(
            "LEANTRADER_DATA_DIR",
            "/app/data",
        )
    )

    target = (
        root
        / "runtime"
        / "engine_registry.json"
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = target.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        target
    )

    return report


def finalize_runtime_intelligence(
    orchestrator,
    logger=None,
):
    logger = (
        logger
        or logging.getLogger(
            "engine_runtime_registry"
        )
    )

    apex, specialists = restore_apex(
        orchestrator,
        logger,
    )

    pre_harmony_heads = (
        collect_engine_heads(
            orchestrator
        )
    )

    harmony, registered = (
        start_harmony(
            orchestrator,
            pre_harmony_heads,
            logger,
        )
    )

    # Allow initial registration values to exist
    # before the first report.
    time.sleep(
        0.05
    )

    report = write_registry(
        orchestrator,
        specialists,
        registered,
    )

    health = report.get(
        "harmony_health",
        {}
    )

    harmony_value = float(
        health.get(
            "harmony",
            0.0,
        )
        or 0.0
    )

    resonance = float(
        health.get(
            "resonance_level",
            0.0,
        )
        or 0.0
    )

    synchronicities = int(
        health.get(
            "synchronicity_count",
            0,
        )
        or 0
    )

    logger.info(
        "🎼 FREQUENCY HARMONY ONLINE "
        "| heads=%s "
        "| registered=%s "
        "| swarm_agents=%s "
        "| harmony=%.4f "
        "| resonance=%.4f "
        "| synchronicities=%s",
        report[
            "unique_heads"
        ],
        registered,
        report[
            "swarm_agents"
        ],
        harmony_value,
        resonance,
        synchronicities,
    )

    if report[
        "protected_duplicates"
    ]:
        raise RuntimeError(
            "Protected duplicate engine "
            "instances remain: "
            + json.dumps(
                report[
                    "protected_duplicates"
                ],
                default=str,
            )
        )

    return report
