"""
CLI-ish tests for graph build init/auto/resume and audit enforcement.
Uses a patched HOME to avoid touching the real filesystem and mocks the LLM client.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click

from commands.graph import build as graph_build


def _write_file(p: Path, content: str = "x = 1\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _setup_project(temp_home: Path, source_dir: Path, name: str = "proj") -> Path:
    """Create a Hound project via ProjectManager under temp HOME."""
    # Late import to use patched HOME
    from commands.project import ProjectManager
    pm = ProjectManager()
    pm.create_project(name, str(source_dir))
    proj_dir = pm.projects_dir / name
    assert proj_dir.exists()
    return proj_dir


def _mock_discovery_and_update(parse: MagicMock):
    """Attach a side-effect that returns a discovery then a simple update."""
    from analysis.graph_builder import GraphDiscovery, GraphUpdate

    def _side_effect(*args, **kwargs):
        schema = kwargs.get('schema') or (len(args) >= 3 and args[2])
        if schema is GraphDiscovery:
            return GraphDiscovery(
                graphs_needed=[
                    {"name": "FooOverview", "focus": "overview"},
                    {"name": "AccessMap", "focus": "roles"},
                    {"name": "ValueFlow", "focus": "value transfer"},
                    {"name": "StateMap", "focus": "state"},
                    {"name": "CallGraph", "focus": "calls"},
                ],
                suggested_node_types=["function", "storage"],
                suggested_edge_types=["calls", "writes", "reads"],
            )
        if schema is GraphUpdate:
            return GraphUpdate(
                target_graph="SystemArchitecture",
                new_nodes=[
                    {"id": "n1", "type": "module", "label": "M1", "refs": []}
                ],
                new_edges=[],
                node_updates=[],
            )
        # Default minimal update
        return GraphUpdate(target_graph="SystemArchitecture")

    parse.side_effect = _side_effect


def test_build_init_and_auto_and_resume(tmp_path, monkeypatch):
    # Patch HOME so ProjectManager writes under tmp
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: temp_home))

    # Create a tiny source repo
    src = tmp_path / "src_demo"
    _write_file(src / "a.py", "def f():\n    return 1\n")

    proj_dir = _setup_project(temp_home, src, name="p1")

    # Mock LLM client
    with patch('analysis.graph_builder.LLMClient') as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        _mock_discovery_and_update(mock_llm.parse)

        # Build with --init
        graph_build(
            project_id="p1",
            config_path=None,
            max_iterations=1,
            max_graphs=1,
            focus_areas=None,
            file_filter=None,
            graph_spec=None,
            refine_existing=True,
            init=True,
            auto=False,
            reuse_ingestion=True,
            visualize=False,
            debug=False,
            quiet=True,
        )

        sys_graph = proj_dir / "graphs" / "graph_SystemArchitecture.json"
        assert sys_graph.exists(), "SystemArchitecture should be created by --init"

        # Build with --auto (should still include SystemArchitecture first and add more graphs)
        from commands import graph as graph_cmd
        with patch.object(graph_cmd.Confirm, 'ask', return_value=True):
            graph_build(
                project_id="p1",
                config_path=None,
                max_iterations=1,
                max_graphs=5,
                focus_areas=None,
                file_filter=None,
                graph_spec=None,
                refine_existing=True,
                init=False,
                auto=True,
                reuse_ingestion=True,
                visualize=False,
                debug=False,
                quiet=True,
            )

        # Expect at least the system graph to still be present
        assert sys_graph.exists()
        # And at least one more graph
        others = list((proj_dir / "graphs").glob("graph_*.json"))
        assert len(others) >= 1

        # Resume: call again; should not error
        with patch.object(graph_cmd.Confirm, 'ask', return_value=True):
            graph_build(
                project_id="p1",
                config_path=None,
                max_iterations=1,
                max_graphs=5,
                focus_areas=None,
                file_filter=None,
                graph_spec=None,
                refine_existing=True,
                init=False,
                auto=True,
                reuse_ingestion=True,
                visualize=False,
                debug=False,
                quiet=True,
            )


def test_audit_enforces_system_architecture(tmp_path, monkeypatch):
    # Patch HOME so ProjectManager writes under tmp
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: temp_home))

    # Create a tiny source repo
    src = tmp_path / "src_demo"
    _write_file(src / "a.py", "def f():\n    return 1\n")

    # Create project and inject a non-system graph only
    from commands.project import ProjectManager
    pm = ProjectManager()
    pm.create_project("p2", str(src))
    proj_dir = pm.projects_dir / "p2"
    gdir = proj_dir / "graphs"
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "graph_Other.json").write_text(json.dumps({
        "name": "Other",
        "internal_name": "Other",
        "nodes": [],
        "edges": [],
        "metadata": {}
    }))

    # AgentRunner should refuse to initialize because SystemArchitecture is missing
    from commands.agent import AgentRunner
    runner = AgentRunner(project_id="p2", config_path=None, iterations=1, time_limit_minutes=1, debug=False, platform=None, model=None)
    ok = runner.initialize()
    assert not ok, "Audit should refuse to start when SystemArchitecture is missing"


def test_init_skips_when_system_exists(tmp_path, monkeypatch):
    # Patch HOME so ProjectManager writes under tmp
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: temp_home))

    # Create a tiny source repo
    src = tmp_path / "src_demo"
    _write_file(src / "a.py", "def f():\n    return 1\n")

    proj_dir = _setup_project(temp_home, src, name="p3")

    # First init to create the SystemArchitecture graph
    with patch('analysis.graph_builder.LLMClient') as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        _mock_discovery_and_update(mock_llm.parse)
        graph_build(
            project_id="p3",
            config_path=None,
            max_iterations=1,
            max_graphs=1,
            focus_areas=None,
            file_filter=None,
            graph_spec=None,
            refine_existing=True,
            init=True,
            auto=False,
            reuse_ingestion=True,
            visualize=False,
            debug=False,
            quiet=True,
        )

    sys_graph = proj_dir / "graphs" / "graph_SystemArchitecture.json"
    assert sys_graph.exists()

    # Second init should skip and not call the LLM at all
    with patch('analysis.graph_builder.LLMClient') as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        try:
            graph_build(
                project_id="p3",
                config_path=None,
                max_iterations=1,
                max_graphs=1,
                focus_areas=None,
                file_filter=None,
                graph_spec=None,
                refine_existing=True,
                init=True,
                auto=False,
                reuse_ingestion=True,
                visualize=False,
                debug=False,
                quiet=True,
            )
        except (SystemExit, click.exceptions.Exit):
            # Typer.Exit maps to click.exceptions.Exit; either is acceptable for a skip path
            pass
        # Ensure no LLM calls occurred during the skipped init
        assert not mock_llm.parse.called
