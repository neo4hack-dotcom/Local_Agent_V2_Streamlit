from __future__ import annotations

from .models import AgentConfig, DatabaseProfile


def _required_engines_for_agent(agent_type: str) -> set[str]:
    if agent_type == "clickhouse_table_manager":
        return {"clickhouse"}
    if agent_type == "sql_analyst":
        return {"clickhouse", "oracle"}
    return set()


def _engine_label(engines: set[str]) -> str:
    if not engines:
        return "any"
    if engines == {"clickhouse", "oracle"}:
        return "ClickHouse/Oracle"
    return "/".join(sorted(engines))


def resolve_database_for_agent(
    *,
    agent: AgentConfig,
    databases: list[DatabaseProfile],
    active_database_id: str | None,
    requested_database_id: str | None,
    required: bool,
) -> DatabaseProfile | None:
    required_engines = _required_engines_for_agent(agent.agent_type) if required else set()

    if not databases:
        if required:
            raise ValueError(
                f"Agent '{agent.name}' requires a database, but no database profiles are configured."
            )
        return None

    by_id = {profile.id: profile for profile in databases}
    cfg = agent.template_config if isinstance(agent.template_config, dict) else {}

    preferred_id_raw = requested_database_id or cfg.get("database_id")
    preferred_id = (
        str(preferred_id_raw).strip() if preferred_id_raw is not None else ""
    )
    if preferred_id:
        selected = by_id.get(preferred_id)
        if selected:
            if required and selected.engine not in required_engines:
                raise ValueError(
                    f"Agent '{agent.name}' requires a {_engine_label(required_engines)} database profile. "
                    f"'{selected.name}' is '{selected.engine}'."
                )
            return selected
        raise ValueError(
            f"Agent '{agent.name}' references unknown database_id '{preferred_id}'."
        )

    preferred_name_raw = cfg.get("database_name")
    preferred_name = (
        str(preferred_name_raw).strip().lower()
        if preferred_name_raw is not None
        else ""
    )
    if preferred_name:
        for profile in databases:
            if profile.name.strip().lower() == preferred_name:
                if required and profile.engine not in required_engines:
                    raise ValueError(
                        f"Agent '{agent.name}' requires a {_engine_label(required_engines)} database profile. "
                        f"'{profile.name}' is '{profile.engine}'."
                    )
                return profile
        raise ValueError(
            f"Agent '{agent.name}' references unknown database_name '{preferred_name_raw}'."
        )

    if not required:
        return None

    if active_database_id and active_database_id in by_id:
        active = by_id[active_database_id]
        if active.engine in required_engines:
            return active

    for profile in databases:
        if profile.engine in required_engines:
            return profile

    if required:
        raise ValueError(
            f"Agent '{agent.name}' requires a database. Set template_config.database_id "
            f"or template_config.database_name to a {_engine_label(required_engines)} profile."
        )

    return None
