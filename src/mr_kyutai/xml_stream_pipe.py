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

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")



print("top of mr_kyutai!!")


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


# Cache: maps original system message text (minus datetime) to converted text.
# Avoids re-converting the same docstrings every turn when only the datetime changed.
_sysmsg_cache: Dict[str, str] = {}
_SYSMSG_CACHE_MAX = 8


def _strip_datetime(text: str) -> str:
    """Strip the datetime line from system message for cache key purposes."""
    import re
    return re.sub(r'^~ \d{4}-\d{2}-\d{2}.*$', '', text, flags=re.MULTILINE).strip()


@pipe(name='process_system_message', priority=5)
async def process_system_message(data: Dict[str, Any], context=None) -> Dict[str, Any]:
    """Convert JSON examples to XML-ish syntax in the system message when xml_streaming is on.

    Takes {'text': system_message}, returns {'text': modified_system_message}.
    Only runs when xml_streaming is enabled. Fast passthrough otherwise.
    Caches by message content (minus datetime) to avoid re-converting every turn.
    """
    print("<<>>> process system message")
    if not _xml_enabled(context):
        print("no sys msg processing")
        return data

    text = data.get('text', '')
    print("pip found but no text!")
    if not text:
        return data

    global _sysmsg_cache
    cache_key = _strip_datetime(text)
    cached = _sysmsg_cache.get(cache_key)
    if cached is not None:
        print("nothing found in proc sys msg cache")
        return {'text': cached}

    converted = convert_docstring_json_examples_to_xml(text)

    print("proc sys msg: converting doc string examples:",len(text))
    
    # Cache the result
    _sysmsg_cache[cache_key] = converted
    if len(_sysmsg_cache) > _SYSMSG_CACHE_MAX:
        keys = list(_sysmsg_cache.keys())
        for k in keys[:_SYSMSG_CACHE_MAX // 2]:
           del _sysmsg_cache[k]

    return {'text': converted}
 
