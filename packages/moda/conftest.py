"""Root pytest configuration for the ``packages/moda`` test run.

TEST-ONLY isolation fix (no product behavior changes).

``test_vapi_e2e.py`` at the package root is not a pytest test module — it defines
no ``test_*`` functions or fixtures. It is a standalone demonstration script that
calls ``moda.init(exporter=ConsoleSpanExporter())`` and
``process_vapi_end_of_call_report(...)`` at MODULE IMPORT time. Because its name
matches pytest's ``test_*`` collection glob, importing it during collection
constructs the process-wide ``TracerWrapper`` singleton (a set-once global) with a
``ConsoleSpanExporter`` attached to the OTel global TracerProvider — BEFORE any
fixture (including the session ``exporter`` fixture) can run. That poisons the
shared singleton for the whole session: the session ``exporter`` fixture's
``Traceloop.init(exporter=InMemory...)`` then no-ops against the already-built
singleton, so every exporter-based test (test_workflows, test_tasks,
test_associations, ...) reads the Console exporter and captures zero spans.

It runs at collection time, so it cannot be fixed with a snapshot/restore fixture
(fixtures run after collection). Exclude the non-test script from collection so
the shared tracing singleton is initialized exactly once, by the session
``exporter`` fixture, with the in-memory exporter the tests assert against.
"""

collect_ignore = ["test_vapi_e2e.py"]
