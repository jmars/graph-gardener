"""Graph Gardener — LLM-powered knowledge graph maintenance.

Two-pass analysis:
  1. CLEANUP — archive stale observations, consolidate entity types, merge duplicates
  2. SYNTHESIS — add missing relations, create summary entities from patterns

Mutations are additive only:
  - Stale observations get ``[archived: YYYY-MM-DD reason]`` appended, never deleted
  - Type renames preserve all observations
  - Merges concatenate observations under one name; self-referencing relations are removed
  - New entities/relations are additive
"""
from __future__ import annotations

import json
import os
import sqlite3
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import llm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", Path.home() / ".vibe" / "memory.db"))

# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_graph(memory_path: Path) -> dict:
    """Load the knowledge graph from SQLite, falling back to JSONL.

    Returns ``{"entities": [...], "relations": [...], "other": []}``.
    """
    if DB_PATH.is_file():
        try:
            return _load_graph_sqlite()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass
    return _load_graph_jsonl(memory_path)


def _load_graph_sqlite() -> dict:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        entities = []
        for row in conn.execute("SELECT name, entity_type FROM entities ORDER BY id"):
            obs_rows = conn.execute(
                "SELECT content FROM observations WHERE entity_id = (SELECT id FROM entities WHERE name = ?) ORDER BY id",
                (row["name"],),
            ).fetchall()
            entities.append({
                "name": row["name"],
                "entityType": row["entity_type"],
                "observations": [o["content"] for o in obs_rows],
            })

        relations = []
        for row in conn.execute("SELECT from_entity, to_entity, relation_type FROM relations ORDER BY id"):
            relations.append({
                "from": row["from_entity"],
                "to": row["to_entity"],
                "relationType": row["relation_type"],
            })
        return {"entities": entities, "relations": relations, "other": []}
    finally:
        conn.close()


def _load_graph_jsonl(memory_path: Path) -> dict:
    entities: list[dict] = []
    relations: list[dict] = []
    other: list[str] = []
    if memory_path.exists():
        for line_num, line in enumerate(
            memory_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                other.append(line)
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed JSON at line {line_num}", file=sys.stderr)
                other.append(line)
                continue
            if item.get("type") == "entity":
                entities.append(item)
            elif item.get("type") == "relation":
                relations.append(item)
            else:
                other.append(line)
    return {"entities": entities, "relations": relations, "other": other}


def save_graph(graph: dict, memory_path: Path) -> None:
    """Write the graph to SQLite (primary) and JSONL (backward compat).

    Uses atomic tmp+rename for JSONL. SQLite writes use a transaction.
    """
    # --- SQLite (primary store) ---
    if DB_PATH.is_file():
        try:
            _save_graph_sqlite(graph)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            print(f"WARNING: SQLite save failed ({e}), falling back to JSONL only", file=sys.stderr)

    # --- JSONL (always written, for backward compat + fst-indexer) ---
    _save_graph_jsonl(graph, memory_path)


def _save_graph_sqlite(graph: dict) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys=ON")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        # Delete all existing data (gardener produces a complete replacement)
        conn.execute("DELETE FROM relations")
        conn.execute("DELETE FROM observations")
        conn.execute("DELETE FROM entities")

        # Insert entities + observations
        for e in graph.get("entities", []):
            cur = conn.execute(
                "INSERT INTO entities (name, entity_type, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (e["name"], e.get("entityType", "unknown"), now, now),
            )
            entity_id = cur.lastrowid
            for obs in e.get("observations", []):
                if isinstance(obs, str):
                    conn.execute(
                        "INSERT INTO observations (entity_id, content, created_at) VALUES (?, ?, ?)",
                        (entity_id, obs, now),
                    )

        # Insert relations
        for r in graph.get("relations", []):
            try:
                conn.execute(
                    "INSERT INTO relations (from_entity, to_entity, relation_type, created_at) VALUES (?, ?, ?, ?)",
                    (r.get("from", ""), r.get("to", ""), r.get("relationType", "related_to"), now),
                )
            except sqlite3.IntegrityError:
                pass

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _save_graph_jsonl(graph: dict, memory_path: Path) -> None:
    lines: list[str] = []
    for e in graph.get("entities", []):
        lines.append(json.dumps({
            "type": "entity",
            "name": e["name"],
            "entityType": e.get("entityType", "unknown"),
            "observations": e.get("observations", []),
        }))
    for r in graph.get("relations", []):
        lines.append(json.dumps({
            "type": "relation",
            "from": r.get("from", ""),
            "to": r.get("to", ""),
            "relationType": r.get("relationType", "related_to"),
        }))
    for raw in graph.get("other", []):
        lines.append(raw.rstrip("\n"))
    data = "\n".join(lines) + "\n"

    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=memory_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, memory_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(graph: dict) -> str:
    """Build the LLM prompt from entity type counts and a sample of entities."""
    entities = graph.get("entities", [])
    relations = graph.get("relations", [])

    # Entity type summary
    type_counts: dict[str, int] = {}
    for e in entities:
        et = e.get("entityType", "unknown")
        type_counts[et] = type_counts.get(et, 0) + 1

    # Relation type summary
    rel_counts: dict[str, int] = {}
    for r in relations:
        rt = r.get("relationType", "unknown")
        rel_counts[rt] = rel_counts.get(rt, 0) + 1

    # Entity detail (all entities with truncated observations)
    entity_detail = []
    for e in entities:
        obs = e.get("observations", [])
        # Truncate long observations for the prompt
        truncated = []
        for o in obs:
            if len(o) > 200:
                truncated.append(o[:197] + "...")
            else:
                truncated.append(o)
        entity_detail.append({
            "name": e["name"],
            "entityType": e.get("entityType", "unknown"),
            "observations": truncated,
        })

    # Relation detail
    relation_detail = [
        {"from": r.get("from", ""), "to": r.get("to", ""), "relationType": r.get("relationType", "?")}
        for r in relations
    ]

    prompt_data = {
        "entity_type_counts": type_counts,
        "relation_type_counts": rel_counts,
        "total_entities": len(entities),
        "total_relations": len(relations),
        "entities": entity_detail,
        "relations": relation_detail,
    }

    return json.dumps(prompt_data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
):
    """Thin wrapper around llm.call — kept for backward compat."""
    return llm.call(prompt, api_url=api_url, api_key=api_key, model=model)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plan(plan: dict) -> list[str]:
    """Validate a mutation plan for structural correctness. Returns warnings."""
    warnings: list[str] = []
    mutations = plan.get("mutations", {})
    if not isinstance(mutations, dict):
        return ["mutations is not a dict"]

    # archive_observations
    for item in mutations.get("archive_observations", []) or []:
        if not isinstance(item, dict):
            warnings.append("archive_observations entry is not a dict")
            continue
        if "entity" not in item:
            warnings.append("archive_observations entry missing 'entity'")

    # rename_types
    for item in mutations.get("rename_types", []) or []:
        if not isinstance(item, dict):
            warnings.append("rename_types entry is not a dict")
            continue
        if "entity" not in item or "new_type" not in item:
            warnings.append("rename_types entry missing 'entity' or 'new_type'")

    # merge_entities
    for item in mutations.get("merge_entities", []) or []:
        if not isinstance(item, dict):
            warnings.append("merge_entities entry is not a dict")
            continue
        if "keep" not in item or "remove" not in item:
            warnings.append("merge_entities entry missing 'keep' or 'remove'")
        elif item["keep"] == item["remove"]:
            warnings.append(f"merge_entities: keep==remove ({item['keep']}) — skipping")

    # add_entities
    for item in mutations.get("add_entities", []) or []:
        if not isinstance(item, dict):
            warnings.append("add_entities entry is not a dict")
            continue
        if "name" not in item or "entityType" not in item:
            warnings.append("add_entities entry missing 'name' or 'entityType'")

    # add_relations
    for item in mutations.get("add_relations", []) or []:
        if not isinstance(item, dict):
            warnings.append("add_relations entry is not a dict")
            continue
        if "from" not in item or "to" not in item:
            warnings.append("add_relations entry missing 'from' or 'to'")

    return warnings


# ---------------------------------------------------------------------------
# Mutation application
# ---------------------------------------------------------------------------

def apply_mutations(graph: dict, plan: dict) -> dict:
    """Apply mutations to the in-memory graph. Returns the modified graph."""
    mutations = plan.get("mutations", {})
    if not isinstance(mutations, dict):
        return graph

    entities = graph.get("entities", [])
    relations = graph.get("relations", [])
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build entity lookup
    entity_map: dict[str, dict] = {e.get("name", ""): e for e in entities}

    # --- archive_observations ---
    for item in mutations.get("archive_observations", []) or []:
        entity_name = item.get("entity", "")
        indices = item.get("indices", [])
        reason = item.get("reason", "archived")
        e = entity_map.get(entity_name)
        if e and indices:
            for idx in sorted(indices, reverse=True):
                if isinstance(idx, int) and 0 <= idx < len(e.get("observations", [])):
                    e["observations"][idx] = f"[archived: {now_iso} {reason}] {e['observations'][idx]}"

    # --- rename_types ---
    for item in mutations.get("rename_types", []) or []:
        entity_name = item.get("entity", "")
        new_type = item.get("new_type", "")
        e = entity_map.get(entity_name)
        if e and new_type:
            e["entityType"] = new_type

    # --- merge_entities ---
    rewired_pairs: set[tuple[str, str]] = set()
    for item in mutations.get("merge_entities", []) or []:
        keep = item.get("keep", "")
        remove = item.get("remove", "")
        if keep == remove or not keep or not remove:
            continue
        keep_e = entity_map.get(keep)
        remove_e = entity_map.get(remove)
        if keep_e and remove_e:
            # Merge observations
            keep_e.setdefault("observations", [])
            keep_e["observations"].extend(remove_e.get("observations", []))
            # Remove merged entity
            entities[:] = [e for e in entities if e.get("name") != remove]
            del entity_map[remove]
            # Rewire relations
            for r in relations:
                if r.get("from") == remove:
                    r["from"] = keep
                    rewired_pairs.add((keep, r.get("to", "")))
                if r.get("to") == remove:
                    r["to"] = keep
                    rewired_pairs.add((r.get("from", ""), keep))
            # Remove self-referencing relations created by rewire
            relations[:] = [
                r for r in relations
                if (r.get("from"), r.get("to")) not in rewired_pairs
                or r.get("from") != r.get("to")
            ]

    # --- add_entities ---
    for item in mutations.get("add_entities", []) or []:
        name = item.get("name", "")
        if name and name not in entity_map:
            new_entity = {
                "name": name,
                "entityType": item.get("entityType", "unknown"),
                "observations": item.get("observations", []),
            }
            entities.append(new_entity)
            entity_map[name] = new_entity

    # --- add_relations ---
    existing_relations = {
        (r.get("from", ""), r.get("to", ""), r.get("relationType", ""))
        for r in relations
    }
    for item in mutations.get("add_relations", []) or []:
        from_e = item.get("from", "")
        to_e = item.get("to", "")
        rtype = item.get("relationType", "related_to")
        if from_e and to_e and (from_e, to_e, rtype) not in existing_relations:
            relations.append({"from": from_e, "to": to_e, "relationType": rtype})
            existing_relations.add((from_e, to_e, rtype))

    return {"entities": entities, "relations": relations, "other": graph.get("other", [])}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    memory_path: Path,
    *,
    apply: bool,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> int:
    """Run graph maintenance: load, prompt, LLM, validate, optionally apply.

    Returns 0 on success, 1 on failure.
    """
    try:
        graph = load_graph(memory_path)
        print(f"Loaded: {len(graph['entities'])} entities, {len(graph['relations'])} relations")

        if not graph["entities"] and not graph["relations"]:
            print("Graph is empty — nothing to do.")
            return 0

        prompt = build_prompt(graph)
        print(f"Prompt: {len(prompt)} chars")

        if len(prompt) > 80_000:
            print(
                f"WARNING: prompt is {len(prompt)} chars — may exceed LLM context window",
                file=sys.stderr,
            )

        resolved_model = model or os.environ.get("GRAPH_GARDENER_MODEL", "deepseek-chat")
        print(f"Sending to {resolved_model}...")

        plan, metadata = call_llm(prompt, api_url=api_url, api_key=api_key, model=model)
        if plan is None:
            print("LLM call failed. No changes made.", file=sys.stderr)
            return 1

        if metadata:
            print(
                f"Model: {metadata.get('model', '?')} | "
                f"Tokens in: {metadata.get('tokens_in', '?')} | "
                f"Tokens out: {metadata.get('tokens_out', '?')}"
            )

        # Validate
        warnings = validate_plan(plan)
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)

        summary = plan.get("summary", "(no summary)")
        mutations = plan.get("mutations", {})
        if isinstance(mutations, dict):
            known_keys = {"archive_observations", "rename_types", "merge_entities",
                          "add_entities", "add_relations"}
            total_mutations = sum(
                len(mutations.get(k, []) or []) for k in known_keys
            )
        else:
            total_mutations = 0

        print(f"\nPlan: {summary}")
        print(f"Mutations: {total_mutations} total")
        for key, val in (mutations or {}).items():
            if val:
                print(f"  {key}: {len(val) if isinstance(val, list) else '?'}")

        if apply:
            # Always create backup with microseconds to prevent collisions
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = memory_path.with_suffix(f".jsonl.bak.{timestamp}")
            shutil.copy2(memory_path, backup_path)
            print(f"Backup saved to: {backup_path}")

            if total_mutations > 0 or warnings:
                graph = apply_mutations(graph, plan)
                save_graph(graph, memory_path)
                print(f"\nSaved: {len(graph['entities'])} entities, {len(graph['relations'])} relations")
            else:
                print("No mutations to apply — file unchanged.")
        else:
            print("\nDry run — no changes applied. Use --apply to commit.")
            if total_mutations > 0:
                print(f"\nFull plan:\n{json.dumps(plan, indent=2)}")

        return 0

    except Exception:  # noqa: BLE001 — top-level safety net for run()
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
