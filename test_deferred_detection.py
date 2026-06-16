#!/usr/bin/env python3
"""Test deferred format detection when [ and <tag are in different chunks."""

import sys
import os
import re
import json

sys.path.insert(0, '/files/mindroot/src')
from mindroot.lib.xml_tool_stream_adapter_v3 import XmlToolStreamAdapter

HYBRID_COMMA_RE = re.compile(r'/>\s*,\s*<')
TRAILING_COMMA_RE = re.compile(r'(/>),\s*$')


def simulate_pipe(chunks, context_agent=None):
    """Simulate the xml_stream_pipe process_stream logic (matching actual code)."""
    state = {}
    if context_agent is None:
        context_agent = {'xml_streaming': True, 'xml_emit_partial_on_chars': 1}
    
    results = []
    
    def on_partial(name, props):
        results.append(('partial', name, dict(props)))
    
    def on_cmd(name, props):
        results.append(('cmd', name, dict(props)))
    
    all_output = []
    
    for i, chunk in enumerate(chunks):
        finish = (i == len(chunks) - 1)
        
        # Handle deferred format detection FIRST
        if 'mode' not in state and state.get('pending_prefix') and not finish:
            prefix = state.pop('pending_prefix')
            combined = prefix + chunk
            chunk = combined
            print(f"  Chunk {i}: re-running detection with combined chunk ({len(chunk)} chars)")
        
        # Format detection on first real chunk
        if 'mode' not in state and not finish:
            stripped = chunk.lstrip()
            if stripped.startswith('['):
                after_bracket = stripped[1:].lstrip()
                if not after_bracket:
                    state['pending_prefix'] = chunk
                    print(f"  Chunk {i}: deferring, buffering")
                    continue
                elif after_bracket.startswith('<'):
                    print(f"  Chunk {i}: hybrid xml mode")
                    state['mode'] = 'xml'
                    state['hybrid_mode'] = True
                    state['hybrid_bracket_stripped'] = True
                    state['prev_chunk_tail'] = ''
                    bracket_idx = chunk.find('[')
                    chunk = chunk[bracket_idx + 1:].lstrip('\n ')
                else:
                    state['mode'] = 'json'
                    print(f"  Chunk {i}: json mode")
            elif stripped.startswith('{'):
                state['mode'] = 'json'
                print(f"  Chunk {i}: json mode")
            else:
                state['mode'] = 'xml'
                print(f"  Chunk {i}: xml mode (pure)")
        
        if state.get('mode') == 'json':
            all_output.append(chunk)
            continue
        
        # Hybrid mode cleaning
        if state.get('hybrid_mode'):
            prev_tail = state.get('prev_chunk_tail', '')
            if prev_tail.rstrip().endswith('>') and chunk.lstrip().startswith(','):
                chunk = chunk.lstrip()
                if chunk.startswith(','):
                    chunk = chunk[1:].lstrip('\n ')
            if prev_tail.rstrip().endswith(',') and chunk.lstrip().startswith('<'):
                chunk = chunk.lstrip('\n ')
            chunk = TRAILING_COMMA_RE.sub(r'\1', chunk)
            chunk = HYBRID_COMMA_RE.sub('/>\n<', chunk)
            if finish:
                chunk = chunk.rstrip()
                if chunk.endswith(']'):
                    chunk = chunk[:-1].rstrip()
            state['prev_chunk_tail'] = chunk[-20:] if len(chunk) > 20 else chunk
        
        # Feed to adapter (create on first use)
        if 'adapter' not in state:
            state['adapter'] = XmlToolStreamAdapter(
                partial_cmd=on_partial,
                cmd=on_cmd,
                speak_command_name='speak',
                emit_partial_on_chars=1,
            )
        
        adapter = state['adapter']
        adapter.feed(chunk)
        if finish:
            adapter.finish()
    
    return results, ''.join(all_output)


# Test 1: [ and <tag in different chunks (the actual bug case)
print("Test 1: [ and <tag in different chunks")
chunks = [
    '[\n',
    '  <tell_and_continue text="Hello!"/>,\n',
    '  <wait_for_user_reply text="OK?"/>\n',
    ']'
]
results, _ = simulate_pipe(chunks)
cmds = [r for r in results if r[0] == 'cmd']
print(f"  Commands: {len(cmds)}")
for c in cmds:
    print(f"    {c[1]}: {c[2]}")
assert len(cmds) == 2, f"Expected 2 commands, got {len(cmds)}"
assert cmds[0][1] == 'tell_and_continue'
assert cmds[1][1] == 'wait_for_user_reply'
print("  PASS")

# Test 2: Pure JSON passthrough
print("\nTest 2: Pure JSON passthrough")
chunks = [
    '[\n',
    '  {"say": {"text": "Hello"}}\n',
    ']'
]
results, json_output = simulate_pipe(chunks)
print(f"  JSON output: {repr(json_output[:50])}")
assert json_output, "Expected JSON passthrough"
print("  PASS")

# Test 3: [ and <tag in same chunk
print("\nTest 3: [ and <tag in same chunk")
chunks = [
    '[<tell_and_continue text="Hello!"/>, <wait_for_user_reply text="OK?"/>]',
]
results, _ = simulate_pipe(chunks)
cmds = [r for r in results if r[0] == 'cmd']
print(f"  Commands: {len(cmds)}")
assert len(cmds) == 2, f"Expected 2 commands, got {len(cmds)}"
print("  PASS")

# Test 4: Pure XML (no brackets)
print("\nTest 4: Pure XML (no brackets)")
chunks = [
    '<tell_and_continue text="Hello!"/>\n',
    '<wait_for_user_reply text="OK?"/>',
]
results, _ = simulate_pipe(chunks)
cmds = [r for r in results if r[0] == 'cmd']
print(f"  Commands: {len(cmds)}")
assert len(cmds) == 2, f"Expected 2 commands, got {len(cmds)}"
print("  PASS")

# Test 5: The exact failing case from the log
print("\nTest 5: Exact failing case from log")
chunks = [
    '[\n',
    '  <tell_and_continue text="Hello! This is a test of the XML format for tell_and_continue."/>,\n',
    '  <tell_and_continue text="Second message to confirm it works with multiple calls."/>,\n',
    '  <wait_for_user_reply text="Did both messages come through?"/>\n',
    ']'
]
results, _ = simulate_pipe(chunks)
cmds = [r for r in results if r[0] == 'cmd']
print(f"  Commands: {len(cmds)}")
for c in cmds:
    print(f"    {c[1]}: {c[2]}")
assert len(cmds) == 3, f"Expected 3 commands, got {len(cmds)}"
assert cmds[0][1] == 'tell_and_continue'
assert cmds[0][2]['text'] == 'Hello! This is a test of the XML format for tell_and_continue.'
assert cmds[1][1] == 'tell_and_continue'
assert cmds[2][1] == 'wait_for_user_reply'
print("  PASS")

print("\nAll tests passed!")
