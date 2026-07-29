"""Tests for graph_gardener.gardener — core graph maintenance logic."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_gardener.gardener import (
    apply_mutations,
    build_prompt,
    call_llm,
    load_graph,
    run,
    save_graph,
    validate_plan,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_memory_file():
    """Create a temporary memory JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"type":"entity","name":"foo","entityType":"concept","observations":["foo obs 1","foo obs 2"]}\n')
        f.write('{"type":"entity","name":"bar","entityType":"technique","observations":["bar obs 1"]}\n')
        f.write('{"type":"relation","from":"foo","to":"bar","relationType":"references"}\n')
        tmp_path = Path(f.name)
    yield tmp_path
    # Cleanup any backup files too
    for p in tmp_path.parent.glob(f"{tmp_path.name}*"):
        p.unlink(missing_ok=True)
    tmp_path.unlink(missing_ok=True)


@pytest.fixture
def empty_memory_file():
    """Create an empty memory file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)
    yield tmp_path
    tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestLoadSave
# ---------------------------------------------------------------------------

class TestLoadSave:
    def test_load_empty_file(self, empty_memory_file):
        graph = load_graph(empty_memory_file)
        assert graph == {"entities": [], "relations": [], "other": []}

    def test_load_entities_and_relations(self, tmp_memory_file):
        graph = load_graph(tmp_memory_file)
        assert len(graph["entities"]) == 2
        assert len(graph["relations"]) == 1
        names = {e["name"] for e in graph["entities"]}
        assert names == {"foo", "bar"}

    def test_save_roundtrip(self, tmp_memory_file):
        graph = load_graph(tmp_memory_file)
        save_graph(graph, tmp_memory_file)
        graph2 = load_graph(tmp_memory_file)
        assert graph == graph2

    def test_save_rejects_non_existent(self, tmp_memory_file):
        """Verify that save raises on a non-existent parent dir."""
        graph = {"entities": [], "relations": [], "other": []}
        with pytest.raises(FileNotFoundError):
            save_graph(graph, Path("/nonexistent/dir/file.jsonl"))

    def test_save_is_atomic(self, tmp_memory_file):
        """If write fails mid-way, original file should be untouched."""
        original_content = tmp_memory_file.read_text()
        graph = {"entities": [], "relations": [], "other": []}

        # Mock os.replace to raise OSError
        with (
            patch("graph_gardener.gardener.os.replace", side_effect=OSError("write failed")),
            pytest.raises(OSError),
        ):
            save_graph(graph, tmp_memory_file)

        # Original file must be unchanged
        assert tmp_memory_file.read_text() == original_content

    def test_load_skips_bad_json_lines(self, tmp_memory_file):
        """Lines that are not valid JSON should be skipped."""
        with open(tmp_memory_file, "a") as f:
            f.write("not valid json\n")
        graph = load_graph(tmp_memory_file)
        assert len(graph["entities"]) == 2
        assert len(graph["relations"]) == 1

    def test_load_skips_unknown_types(self, tmp_memory_file):
        """Lines with unknown types should be preserved in 'other'."""
        with open(tmp_memory_file, "a") as f:
            f.write('{"type":"unknown","data":"stuff"}\n')
        graph = load_graph(tmp_memory_file)
        assert len(graph["entities"]) == 2
        assert len(graph["relations"]) == 1
        assert len(graph["other"]) == 1
        assert "unknown" in graph["other"][0]

    def test_load_preserves_other_lines(self, tmp_memory_file):
        """Non-entity/relation lines survive a save/load round-trip."""
        with open(tmp_memory_file, "a") as f:
            f.write('{"type":"metadata","version":1}\n')
            f.write('{"type":"session","id":"abc"}\n')
        graph = load_graph(tmp_memory_file)
        assert len(graph["other"]) == 2
        save_graph(graph, tmp_memory_file)
        graph2 = load_graph(tmp_memory_file)
        assert len(graph2["other"]) == 2


# ---------------------------------------------------------------------------
# TestBuildPrompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_prompt_contains_entity_names(self):
        graph = {
            "entities": [
                {"name": "Alice", "entityType": "person", "observations": ["likes cats"]},
                {"name": "Bob", "entityType": "person", "observations": ["likes dogs"]},
            ],
            "relations": [],
        }
        prompt = build_prompt(graph)
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "[person]" in prompt

    def test_prompt_contains_relation(self):
        graph = {
            "entities": [
                {"name": "Alice", "entityType": "person", "observations": []},
                {"name": "Bob", "entityType": "person", "observations": []},
            ],
            "relations": [
                {"from": "Alice", "to": "Bob", "relationType": "knows"},
            ],
        }
        prompt = build_prompt(graph)
        assert "Alice --[knows]--> Bob" in prompt

    def test_empty_graph(self):
        graph = {"entities": [], "relations": []}
        prompt = build_prompt(graph)
        assert "Entities (0):" in prompt
        assert "Relations (0):" in prompt

    def test_observation_truncation(self):
        """Observations longer than 150 chars should be truncated."""
        long_obs = "x" * 300
        graph = {
            "entities": [
                {"name": "Test", "entityType": "test", "observations": [long_obs]},
            ],
            "relations": [],
        }
        prompt = build_prompt(graph)
        assert "x" * 150 in prompt
        assert "x" * 151 not in prompt


# ---------------------------------------------------------------------------
# TestValidatePlan
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan_no_warnings(self):
        plan = {
            "summary": "cleanup",
            "mutations": {
                "archive_observations": [{"entity": "foo", "observation_index": 0, "reason": "stale"}],
                "rename_types": [{"entity": "foo", "new_type": "concept"}],
                "merge_entities": [{"keep": "foo", "remove": "bar"}],
                "add_relations": [{"from": "foo", "to": "bar", "relationType": "references"}],
                "add_entities": [{"name": "baz", "entityType": "summary", "observations": ["obs1"]}],
            },
        }
        warnings = validate_plan(plan)
        assert warnings == []

    def test_missing_mutations_key(self):
        plan = {"summary": "cleanup"}
        warnings = validate_plan(plan)
        assert "missing 'mutations' key" in warnings[0]

    def test_mutations_not_dict(self):
        plan = {"mutations": "not-a-dict"}
        warnings = validate_plan(plan)
        assert any("must be a dict" in w for w in warnings)

    def test_wrong_type_in_archive(self):
        """Missing required key in archive_observations should warn."""
        plan = {
            "mutations": {
                "archive_observations": [{"entity": "foo"}],  # missing observation_index, reason
            },
        }
        warnings = validate_plan(plan)
        assert any("missing required key" in w for w in warnings)

    def test_invalid_item_type(self):
        """Non-dict items should warn."""
        plan = {
            "mutations": {
                "add_entities": ["not-a-dict"],
            },
        }
        warnings = validate_plan(plan)
        assert any("is not a dict" in w for w in warnings)

    def test_not_a_list_in_mutation_key(self):
        """Mutation key that isn't a list should warn."""
        plan = {
            "mutations": {
                "add_relations": "not-a-list",
            },
        }
        warnings = validate_plan(plan)
        assert any("must be a list" in w for w in warnings)


# ---------------------------------------------------------------------------
# TestApplyMutations
# ---------------------------------------------------------------------------

class TestApplyMutations:
    def test_archive_observation(self):
        graph = {
            "entities": [
                {"name": "foo", "entityType": "concept", "observations": ["obs1", "obs2"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "archive_observations": [{"entity": "foo", "observation_index": 0, "reason": "stale ref"}],
            },
        }
        result = apply_mutations(graph, plan)
        obs = result["entities"][0]["observations"][0]
        assert "[archived:" in obs
        assert "stale ref" in obs

    def test_archive_out_of_bounds(self):
        graph = {
            "entities": [
                {"name": "foo", "entityType": "concept", "observations": ["obs1"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "archive_observations": [{"entity": "foo", "observation_index": 99, "reason": "test"}],
            },
        }
        result = apply_mutations(graph, plan)
        # No change to observations
        assert result["entities"][0]["observations"] == ["obs1"]

    def test_rename_type(self):
        graph = {
            "entities": [
                {"name": "foo", "entityType": "Technique", "observations": []},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "rename_types": [{"entity": "foo", "new_type": "convention"}],
            },
        }
        result = apply_mutations(graph, plan)
        assert result["entities"][0]["entityType"] == "convention"

    def test_rename_nonexistent_entity(self):
        graph = {
            "entities": [
                {"name": "foo", "entityType": "Technique", "observations": []},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "rename_types": [{"entity": "nonexistent", "new_type": "convention"}],
            },
        }
        result = apply_mutations(graph, plan)
        assert result["entities"][0]["entityType"] == "Technique"

    def test_merge_entities(self):
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["alice obs"]},
                {"name": "alice_dup", "entityType": "person", "observations": ["dup obs"]},
            ],
            "relations": [
                {"from": "alice_dup", "to": "bob", "relationType": "knows"},
                {"from": "charlie", "to": "alice_dup", "relationType": "knows"},
            ],
        }
        plan = {
            "mutations": {
                "merge_entities": [{"keep": "alice", "remove": "alice_dup"}],
            },
        }
        result = apply_mutations(graph, plan)
        names = {e["name"] for e in result["entities"]}
        assert "alice" in names
        assert "alice_dup" not in names
        # Observations combined
        alice = next(e for e in result["entities"] if e["name"] == "alice")
        assert "alice obs" in alice["observations"]
        assert "dup obs" in alice["observations"]
        # Relations rewired
        rels = result["relations"]
        assert any(r["from"] == "alice" and r["to"] == "bob" for r in rels)
        assert any(r["from"] == "charlie" and r["to"] == "alice" for r in rels)

    def test_merge_deduplicates_observations(self):
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["common obs", "unique obs"]},
                {"name": "alice_dup", "entityType": "person", "observations": ["common obs"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "merge_entities": [{"keep": "alice", "remove": "alice_dup"}],
            },
        }
        result = apply_mutations(graph, plan)
        alice = next(e for e in result["entities"] if e["name"] == "alice")
        assert alice["observations"] == ["common obs", "unique obs"]

    def test_merge_removes_self_references(self):
        """If A has relation to B and B is merged into A, the resulting A→A relation is removed."""
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["obs"]},
                {"name": "bob", "entityType": "person", "observations": ["obs"]},
            ],
            "relations": [
                {"from": "alice", "to": "bob", "relationType": "knows"},
            ],
        }
        plan = {
            "mutations": {
                "merge_entities": [{"keep": "alice", "remove": "bob"}],
            },
        }
        result = apply_mutations(graph, plan)
        # Should have no self-referencing relations
        for r in result["relations"]:
            assert r["from"] != r["to"], f"Self-referencing relation found: {r}"
        assert len(result["relations"]) == 0

    def test_add_relations(self):
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": []},
                {"name": "bob", "entityType": "person", "observations": []},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "add_relations": [{"from": "alice", "to": "bob", "relationType": "knows"}],
            },
        }
        result = apply_mutations(graph, plan)
        assert len(result["relations"]) == 1
        assert result["relations"][0]["from"] == "alice"
        assert result["relations"][0]["to"] == "bob"
        assert result["relations"][0]["relationType"] == "knows"

    def test_add_relations_skips_duplicate(self):
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": []},
                {"name": "bob", "entityType": "person", "observations": []},
            ],
            "relations": [
                {"from": "alice", "to": "bob", "relationType": "knows"},
            ],
        }
        plan = {
            "mutations": {
                "add_relations": [{"from": "alice", "to": "bob", "relationType": "knows"}],
            },
        }
        result = apply_mutations(graph, plan)
        assert len(result["relations"]) == 1

    def test_add_entities_before_relations(self):
        """A plan that adds entity X and a relation to X in the same batch works."""
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": []},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "add_entities": [
                    {"name": "bob", "entityType": "person", "observations": ["bob obs"]},
                ],
                "add_relations": [
                    {"from": "alice", "to": "bob", "relationType": "knows"},
                ],
            },
        }
        result = apply_mutations(graph, plan)
        names = {e["name"] for e in result["entities"]}
        assert "bob" in names
        assert len(result["relations"]) == 1

    def test_add_entities_skips_existing(self):
        """Adding entity with existing name does nothing."""
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["original"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "add_entities": [
                    {"name": "alice", "entityType": "robot", "observations": ["new data"]},
                ],
            },
        }
        result = apply_mutations(graph, plan)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["entityType"] == "person"  # not overwritten

    def test_no_mutations_key(self):
        """Plan without mutations key should be a no-op."""
        graph = {
            "entities": [
                {"name": "foo", "entityType": "concept", "observations": ["obs"]},
            ],
            "relations": [],
        }
        plan = {"summary": "nothing"}
        result = apply_mutations(graph, plan)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["observations"] == ["obs"]

    def test_self_merge_is_skipped(self):
        """A merge where keep == remove should be skipped, not delete the entity."""
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["obs"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "merge_entities": [{"keep": "alice", "remove": "alice"}],
            },
        }
        result = apply_mutations(graph, plan)
        names = {e["name"] for e in result["entities"]}
        assert "alice" in names
        assert len(result["entities"]) == 1

    def test_malformed_plan_does_not_crash(self):
        """Plan with wrong types, missing keys, or string indices should not crash."""
        graph = {
            "entities": [
                {"name": "foo", "entityType": "concept", "observations": ["obs"]},
            ],
            "relations": [],
        }
        plan = {
            "mutations": {
                "archive_observations": [
                    {"entity": "foo", "observation_index": "not_an_int", "reason": "test"},
                    {"reason": "only_reason"},  # missing entity
                    "not_a_dict",  # not even a dict
                ],
                "rename_types": [
                    {"entity": "foo"},  # missing new_type
                    123,  # not a dict
                ],
                "merge_entities": [
                    {"keep": "foo"},  # missing remove
                ],
                "add_entities": [
                    {"observations": "not_a_list"},  # missing name
                ],
                "add_relations": [
                    {"from": "foo", "relationType": "knows"},  # missing to
                ],
            },
        }
        # Should not raise
        result = apply_mutations(graph, plan)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "foo"

    def test_existing_self_references_preserved(self):
        """Self-references that existed before a merge should be preserved."""
        graph = {
            "entities": [
                {"name": "alice", "entityType": "person", "observations": ["obs"]},
                {"name": "bob", "entityType": "person", "observations": ["obs"]},
            ],
            "relations": [
                {"from": "alice", "to": "alice", "relationType": "self_knows"},
            ],
        }
        plan = {
            "mutations": {
                "merge_entities": [{"keep": "alice", "remove": "bob"}],
            },
        }
        result = apply_mutations(graph, plan)
        # The pre-existing alice→alice self-reference should be preserved
        assert any(
            r["from"] == "alice" and r["to"] == "alice" and r["relationType"] == "self_knows"
            for r in result["relations"]
        )


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------

class TestRun:
    def test_dry_run_does_not_modify_file(self, tmp_memory_file):
        original = tmp_memory_file.read_text()
        with patch("graph_gardener.gardener.call_llm") as mock_call:
            mock_call.return_value = (
                {"summary": "test", "mutations": {}},
                {"model": "test", "tokens_in": 0, "tokens_out": 0, "generated_at": "now"},
            )
            rc = run(tmp_memory_file, apply=False)

        assert rc == 0
        assert tmp_memory_file.read_text() == original

    def test_apply_creates_backup(self, tmp_memory_file):
        plan = {
            "summary": "rename type",
            "mutations": {
                "rename_types": [{"entity": "foo", "new_type": "updated"}],
            },
        }
        with patch("graph_gardener.gardener.call_llm") as mock_call:
            mock_call.return_value = (
                plan,
                {"model": "test", "tokens_in": 0, "tokens_out": 0, "generated_at": "now"},
            )
            rc = run(tmp_memory_file, apply=True)

        assert rc == 0
        # Backup file should exist
        backups = list(tmp_memory_file.parent.glob(f"{tmp_memory_file.name}.bak.*"))
        assert len(backups) >= 1

    def test_apply_modifies_file(self, tmp_memory_file):
        """When mutations are applied, the file should change."""
        plan = {
            "summary": "rename type",
            "mutations": {
                "rename_types": [{"entity": "foo", "new_type": "updated-type"}],
            },
        }
        with patch("graph_gardener.gardener.call_llm") as mock_call:
            mock_call.return_value = (
                plan,
                {"model": "test", "tokens_in": 10, "tokens_out": 5, "generated_at": "now"},
            )
            rc = run(tmp_memory_file, apply=True)

        assert rc == 0
        # Reload and verify
        graph = load_graph(tmp_memory_file)
        foo = next(e for e in graph["entities"] if e["name"] == "foo")
        assert foo["entityType"] == "updated-type"

    def test_llm_failure_returns_1(self, tmp_memory_file):
        with patch("graph_gardener.gardener.call_llm") as mock_call:
            mock_call.return_value = (None, None)
            rc = run(tmp_memory_file, apply=False)

        assert rc == 1

    def test_empty_graph_skips_llm(self):
        """Empty graph should return 0 without calling LLM."""
        with (
            patch("graph_gardener.gardener.load_graph", return_value={"entities": [], "relations": [], "other": []}),
            patch("graph_gardener.gardener.call_llm") as mock_call,
        ):
            rc = run(Path("/nonexistent/file.jsonl"), apply=False)
            mock_call.assert_not_called()
        assert rc == 0

    def test_unhandled_exception_returns_1(self, tmp_memory_file):
        """If an unexpected exception occurs, run should return 1."""
        with patch("graph_gardener.gardener.load_graph", side_effect=RuntimeError("disk error")):
            rc = run(tmp_memory_file, apply=False)
        assert rc == 1


# ---------------------------------------------------------------------------
# TestCallLlm
# ---------------------------------------------------------------------------

class TestCallLlm:
    def test_call_llm_success(self):
        """Verify call_llm wraps llm.call correctly."""
        mock_result = {"mutations": {"add_entities": []}}
        mock_metadata = {"model": "test", "tokens_in": 5, "tokens_out": 3, "generated_at": "now"}

        with patch("graph_gardener.gardener.llm.call") as mock_llm_call:
            mock_llm_call.return_value = (mock_result, mock_metadata)
            result, metadata = call_llm("test prompt", api_url="https://x.com/v1", api_key="k", model="m")

        assert result == mock_result
        assert metadata == mock_metadata

    def test_call_llm_failure(self):
        with patch("graph_gardener.gardener.llm.call") as mock_llm_call:
            mock_llm_call.return_value = (None, None)
            result, metadata = call_llm("test prompt", api_url="https://x.com/v1", api_key="k")

        assert result is None
        assert metadata is None
