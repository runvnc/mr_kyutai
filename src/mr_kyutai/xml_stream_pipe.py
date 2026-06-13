"""
process_stream pipe for mr_kyutai: transforms XML-ish LLM output into JSON
command arrays that feed into the normal parse_cmd_stream parser.

Raw text outside XML tags becomes speak commands. Tool tags become their
respective JSON commands.

For streaming speech, the pipe outputs partial JSON that the parser handles
as partial commands, enabling low-latency TTS via partial_command('speak', ...).

Activation:
    - Agent config: context.agent.get('xml_streaming') is truthy
    - Environment: MR_XML_STREAMING=1

No side effects: all output is text (JSON arrays) that the parser handles.
"""

import json
import os
from typing import Any, Dict

from lib.pipelines.pipe import pipe
from lib.xml_tool_stream_adapter_v3 import XmlToolStreamAdapter


def _xml_enabled(context) -> bool:
    """Check if XML streaming mode is enabled."""
    if os.environ.get('MR_XML_STREAMING') == '1':
        return True
    if context is not None and hasattr(context, 'agent') and context.agent:
        if context.agent.get('xml_streaming'):
            return True
    return False


def _json_str_content(s: str) -> str:
    """Return the JSON-escaped content of a string (no surrounding quotes).

    For use when building JSON source text piece by piece inside a string value.
    """
    return json.dumps(s)[1:-1]


def _init_state(state: Dict[str, Any], context):
    """Initialize XML stream processing state and adapter callbacks."""
    state['mode'] = 'xml'
    state['output_pieces'] = []
    state['speak_json_open'] = False
    state['last_spoken_len'] = 0

    emit_chars = 8
    if context is not None and hasattr(context, 'agent') and context.agent:
        emit_chars = int(context.agent.get('xml_emit_partial_on_chars', emit_chars))

    def on_partial(name, props):
        """Adapter callback: speech text has grown. Output as growing speak JSON."""
        if name == 'speak':
            text = props.get('text', '')
            new_len = len(text)
            old_len = state['last_spoken_len']
            delta = text[old_len:]
            if not delta:
                return
            if not state['speak_json_open']:
                # Start a new speak command as partial JSON
                state['output_pieces'].append('[{"speak": {"text": "')
                state['output_pieces'].append(_json_str_content(text))
                state['speak_json_open'] = True
            else:
                # Continue the existing speak command with just the delta
                state['output_pieces'].append(_json_str_content(delta))
            state['last_spoken_len'] = new_len

    def on_cmd(name, props):
        """Adapter callback: a tool command is complete. Close any open speak and output the tool."""
        if state['speak_json_open']:
            state['output_pieces'].append('"}}]')
            state['speak_json_open'] = False
            state['last_spoken_len'] = 0
        state['output_pieces'].append(json.dumps([{name: props}]))

    state['adapter'] = XmlToolStreamAdapter(
        partial_cmd=on_partial,
        cmd=on_cmd,
        speak_command_name='speak',
        emit_partial_on_chars=emit_chars,
    )


@pipe(name='process_stream', priority=5)
async def process_stream(data: Dict[str, Any], context=None) -> Dict[str, Any]:
    """
    Transform XML-ish LLM stream chunks into JSON command arrays.

    Takes {'chunk': text, 'finish': bool}, returns {'chunk': modified_text}.

    - Raw text outside XML tags becomes [{"speak": {"text": "..."}}]
    - Tool tags become their JSON command equivalents
    - If the stream starts with '[' or '{', assumes JSON and passes through
    - No side effects: all output is text that the parser handles
    """
    chunk = data.get('chunk', '')
    finish = data.get('finish', False)

    if not _xml_enabled(context):
        return data

    if context is None:
        return data

    state = context.data.setdefault('_xml_stream_state', {})

    # Format detection on first real chunk
    if 'mode' not in state and not finish:
        stripped = chunk.lstrip()
        if stripped.startswith('[') or stripped.startswith('{'):
            state['mode'] = 'json'
        else:
            _init_state(state, context)

    if state.get('mode') == 'json':
        return data

    adapter = state.get('adapter')
    if adapter is None:
        return data

    try:
        if finish:
            adapter.finish()
        else:
            adapter.feed(chunk)
    except Exception:
        # If adapter blows up, fall back to passthrough
        return data

    # Close any open speak command on finish
    if finish and state.get('speak_json_open'):
        state['output_pieces'].append('"}}]')
        state['speak_json_open'] = False

    output = ''.join(state['output_pieces'])
    state['output_pieces'] = []

    if output:
        return {'chunk': output}
    return {'chunk': ''}


# ── System message docstring conversion ──────────────────────────────────

from lib.xml_docstring_adapter import convert_docstring_json_examples_to_xml

# Per-docstring conversion cache: {original_docstring: converted_docstring}
_docstring_cache: Dict[str, str] = {}
_DOCSTRING_CACHE_MAX = 256


def _convert_docstrings(data: Dict[str, Any], context) -> Dict[str, Any]:
    """Convert command docstrings from JSON examples to XML-ish syntax.

    Only runs when xml_streaming is enabled. Caches per-docstring so we
    don't re-convert the same docstrings every turn.
    """
    if not _xml_enabled(context):
        return data

    command_docs = data.get('command_docs')
    if not command_docs or not isinstance(command_docs, dict):
        return data

    global _docstring_cache
    converted = {}
    for cmd_name, docstring in command_docs.items():
        if not isinstance(docstring, str):
            converted[cmd_name] = docstring
            continue
        cached = _docstring_cache.get(docstring)
        if cached is not None:
            converted[cmd_name] = cached
        else:
            result = convert_docstring_json_examples_to_xml(docstring)
            _docstring_cache[docstring] = result
            converted[cmd_name] = result

    # Evict if cache is too large
    if len(_docstring_cache) > _DOCSTRING_CACHE_MAX:
        keys = list(_docstring_cache.keys())
        for k in keys[:_DOCSTRING_CACHE_MAX // 2]:
            del _docstring_cache[k]

    data['command_docs'] = converted
    return data


@pipe(name='process_system_data', priority=5)
async def process_system_data(data: Dict[str, Any], context=None) -> Dict[str, Any]:
    """Convert command docstrings from JSON to XML-ish syntax when xml_streaming is on.

    Takes the data dict passed to the system template, returns modified data.
    Caches per-docstring so conversion only runs once per unique docstring.
    """
    return _convert_docstrings(data, context)
