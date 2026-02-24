from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_settings_repo
from app.core.db_connectors import connector_for
from app.core.models import (
    AppSettings,
    DatabaseConfigExport,
    DatabaseConfigImportRequest,
    DatabaseConfigImportResponse,
    DatabaseProfile,
    DatabaseProfileCreate,
    DatabaseProfileUpdate,
)
from app.core.storage import JSONRepository

router = APIRouter(prefix="/databases", tags=["databases"])


def _find_database(settings: AppSettings, database_id: str) -> DatabaseProfile:
    for profile in settings.databases:
        if profile.id == database_id:
            return profile
    raise HTTPException(status_code=404, detail="Database profile not found.")


def _validate_unique_database_ids(databases: list[DatabaseProfile]) -> None:
    seen: set[str] = set()
    for profile in databases:
        if profile.id in seen:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate database id in import payload: '{profile.id}'.",
            )
        seen.add(profile.id)


@router.get("", response_model=list[DatabaseProfile])
def list_databases(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> list[DatabaseProfile]:
    return settings_repo.load().databases


@router.post("", response_model=DatabaseProfile, status_code=201)
def create_database(
    payload: DatabaseProfileCreate,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> DatabaseProfile:
    settings = settings_repo.load()
    profile = DatabaseProfile(id=uuid4().hex, **payload.model_dump())
    settings.databases.append(profile)
    if not settings.active_database_id:
        settings.active_database_id = profile.id
    settings_repo.save(settings)
    return profile


@router.put("/{database_id}", response_model=DatabaseProfile)
def update_database(
    database_id: str,
    payload: DatabaseProfileUpdate,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> DatabaseProfile:
    settings = settings_repo.load()
    for index, profile in enumerate(settings.databases):
        if profile.id == database_id:
            updated = DatabaseProfile(id=database_id, **payload.model_dump())
            settings.databases[index] = updated
            settings_repo.save(settings)
            return updated
    raise HTTPException(status_code=404, detail="Database profile not found.")


@router.delete("/{database_id}", status_code=204)
def delete_database(
    database_id: str,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> None:
    settings = settings_repo.load()
    settings.databases = [db for db in settings.databases if db.id != database_id]
    if settings.active_database_id == database_id:
        settings.active_database_id = settings.databases[0].id if settings.databases else None
    settings_repo.save(settings)


@router.put("/active/{database_id}", response_model=AppSettings)
def set_active_database(
    database_id: str,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> AppSettings:
    settings = settings_repo.load()
    _find_database(settings, database_id)
    settings.active_database_id = database_id
    settings_repo.save(settings)
    return settings


@router.post("/{database_id}/test")
def test_database_connection(
    database_id: str,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> dict:
    settings = settings_repo.load()
    profile = _find_database(settings, database_id)

    try:
        return connector_for(profile).test_connection()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/export", response_model=DatabaseConfigExport)
def export_databases_config(
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> DatabaseConfigExport:
    settings = settings_repo.load()
    return DatabaseConfigExport(
        exported_at=datetime.now(timezone.utc).isoformat(),
        active_database_id=settings.active_database_id,
        databases=[profile.model_copy(deep=True) for profile in settings.databases],
    )


@router.post("/import", response_model=DatabaseConfigImportResponse)
def import_databases_config(
    payload: DatabaseConfigImportRequest,
    settings_repo: JSONRepository[AppSettings] = Depends(get_settings_repo),
) -> DatabaseConfigImportResponse:
    imported_databases = [profile.model_copy(deep=True) for profile in payload.payload.databases]
    _validate_unique_database_ids(imported_databases)

    settings = settings_repo.load()
    imported_count = len(imported_databases)

    if payload.mode == "replace":
        imported_ids = {profile.id for profile in imported_databases}
        settings.databases = imported_databases

        requested_active = payload.payload.active_database_id
        if requested_active and requested_active in imported_ids:
            settings.active_database_id = requested_active
        else:
            settings.active_database_id = imported_databases[0].id if imported_databases else None

        settings_repo.save(settings)
        return DatabaseConfigImportResponse(
            mode="replace",
            imported_databases=imported_count,
            created_databases=0,
            updated_databases=0,
            replaced_databases=imported_count,
            active_database_id=settings.active_database_id,
        )

    existing_ids = {profile.id for profile in settings.databases}
    created = 0
    updated = 0

    merged_by_id: dict[str, DatabaseProfile] = {profile.id: profile for profile in settings.databases}
    ordered_ids = [profile.id for profile in settings.databases]
    for profile in imported_databases:
        if profile.id in existing_ids:
            updated += 1
        else:
            created += 1
            ordered_ids.append(profile.id)
        merged_by_id[profile.id] = profile

    settings.databases = [merged_by_id[profile_id] for profile_id in ordered_ids if profile_id in merged_by_id]

    requested_active = payload.payload.active_database_id
    all_ids = {profile.id for profile in settings.databases}
    if requested_active and requested_active in all_ids:
        settings.active_database_id = requested_active
    elif settings.active_database_id not in all_ids:
        settings.active_database_id = settings.databases[0].id if settings.databases else None

    settings_repo.save(settings)
    return DatabaseConfigImportResponse(
        mode="merge",
        imported_databases=imported_count,
        created_databases=created,
        updated_databases=updated,
        replaced_databases=0,
        active_database_id=settings.active_database_id,
    )
