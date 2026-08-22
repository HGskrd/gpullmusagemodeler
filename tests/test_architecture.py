"""The module dependency graph, enforced.

The refactor that produced this layout removed three import cycles and the 13
lazy (function-level) imports that were hiding them. Both are easy to
reintroduce and neither shows up as a test failure anywhere else, so they are
pinned here.

The layer order is the one documented in README.md's "Module Layout" table.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Lower number = lower layer. A module may import from its own layer or below.
LAYERS = {
    "data": 0,
    "deployment": 0,
    "cloud_policy": 0,
    "use_case_evidence": 0,
    "state": 1,
    "scenarios": 1,
    "tracking": 1,
    "calc": 2,
    "placement": 2,
    "engine": 2,
    "planner_service": 3,
    "presentation": 4,
    "web": 5,
    "app": 6,
}

# state.py still calls three estimator functions (avg_dist, resolve_spec_runtime,
# valid_strategies) from inside its mutators, plus the EfficiencyParams value
# object. Untangling it means moving those calls up into planner_service. Until
# then this is the one accepted upward edge -- do not add others.
ACCEPTED_UPWARD_EDGES = {("state", "calc")}


def _source_files():
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts
        if parts[0] in {"tests", "build", "dist"} or parts[0].startswith("."):
            continue
        if "__pycache__" in parts:
            continue
        yield relative, path


def _dependency_graph():
    files = list(_source_files())
    top_level = {relative.parts[0].removesuffix(".py") for relative, _ in files}
    graph: dict[str, set[str]] = {}
    for relative, path in files:
        owner = relative.parts[0].removesuffix(".py")
        deps = graph.setdefault(owner, set())
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in top_level:
                        deps.add(root)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root in top_level:
                    deps.add(root)
        deps.discard(owner)
    return graph


class DependencyGraphTests(unittest.TestCase):
    def test_the_import_graph_is_acyclic(self):
        graph = _dependency_graph()
        state = dict.fromkeys(graph, "unvisited")
        cycles = []

        def visit(node, path):
            state[node] = "open"
            for child in sorted(graph.get(node, ())):
                if state.get(child, "unvisited") == "open":
                    cycles.append(" -> ".join(path + [node, child]))
                elif state.get(child, "unvisited") == "unvisited":
                    visit(child, path + [node])
            state[node] = "done"

        for node in sorted(graph):
            if state[node] == "unvisited":
                visit(node, [])
        self.assertEqual(cycles, [], f"import cycle(s) reintroduced: {cycles}")

    def test_no_module_imports_from_a_higher_layer(self):
        graph = _dependency_graph()
        violations = sorted(
            (module, dependency)
            for module, dependencies in graph.items()
            for dependency in dependencies
            if module in LAYERS
            and dependency in LAYERS
            and LAYERS[dependency] > LAYERS[module]
            and (module, dependency) not in ACCEPTED_UPWARD_EDGES
        )
        self.assertEqual(violations, [], f"upward dependencies: {violations}")

    def test_the_accepted_upward_edges_still_exist(self):
        """Delete an exemption once the debt it records is paid off."""
        graph = _dependency_graph()
        stale = sorted(
            edge for edge in ACCEPTED_UPWARD_EDGES if edge[1] not in graph.get(edge[0], ())
        )
        self.assertEqual(
            stale,
            [],
            "ACCEPTED_UPWARD_EDGES lists a dependency that no longer exists; remove it",
        )

    def test_app_is_only_a_composition_root(self):
        graph = _dependency_graph()
        self.assertEqual(
            graph["app"],
            {"presentation", "web"},
            "app.py should wire blueprints and filters, not reach into the engine",
        )

    def test_no_route_handlers_live_outside_the_web_package(self):
        for relative, path in _source_files():
            if relative.parts[0] == "web":
                continue
            with self.subTest(module=str(relative)):
                self.assertNotIn(
                    "_bp.route(",
                    path.read_text(encoding="utf-8"),
                    f"{relative} owns routes; they belong in web/",
                )


class LazyImportTests(unittest.TestCase):
    def test_no_function_level_imports_outside_the_allowed_set(self):
        """Lazy imports are how the old cycles were hidden. Keep them out."""
        allowed_roots = {"typing", "dataclasses", "__future__"}
        offenders = []
        for relative, path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Import):
                        roots = [a.name.split(".")[0] for a in inner.names]
                    elif isinstance(inner, ast.ImportFrom):
                        roots = [(inner.module or "").split(".")[0]]
                    else:
                        continue
                    for root in roots:
                        if root not in allowed_roots:
                            offenders.append(f"{relative}:{inner.lineno} imports {root}")
        self.assertEqual(
            sorted(offenders),
            [],
            "function-level imports reintroduced; they hide import cycles",
        )


if __name__ == "__main__":
    unittest.main()
