"""
DEPRECATED / REMOVED.

XML / raw-text command streaming is now a first-class capability in MindRoot
core (see lib/xml_stream_events.py, coreplugins/agent/agent.py
parse_xml_cmd_stream, and templates/system_xml.jinja2), gated per-agent by
MR_XML_STREAMING in the agent's env overrides (context.env).

This module USED TO register two pipes:
  - process_stream            (XML-ish chunks -> JSON command arrays)
  - process_system_message    (rebuild a compact system message)

Both moved to core. They are intentionally NOT registered here anymore.

Why this matters (performance): PipelineManager.execute_pipeline short-circuits
for free when a pipeline name has ZERO registered pipes ('if name not in
self.pipes: return data'). The agent's JSON streaming loop calls the
'process_stream' pipeline once PER TOKEN. A registered no-op pipe forced an
inspect.iscoroutinefunction() check + coroutine creation + await on every single
token for every JSON-mode agent that had mr_kyutai loaded. Registering nothing
restores the zero-cost dict-miss path.

mr_kyutai now only provides TTS: the realtime partial_command 'speak' streaming
pipe and the speak command (see realtime_stream.py and mod.py).

This file is kept only so any lingering 'from . import xml_stream_pipe' import
does not break; it has no side effects. Safe to delete once nothing imports it.
"""

# Intentionally empty: no pipe registrations.
