#!/usr/bin/env python3
"""Test hybrid format detection and handling in xml_stream_pipe."""

import sys
import os
import re
import json

# Add paths
TRAILING_COMMA_RE = re.compile(r'(/>),\s*$')
sys.path.insert(0, '/files/mindroot/src')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mindroot.lib.xml_tool_stream_adapter_v3 import XmlToolStreamAdapter
HYBRID_COMMA_RE = re.compile(r'/>\s*,\s*<')
TRAILING_COMMA_RE = re.compile(r'(/>),\s*$')


def test_hybrid_comma_regex():
    """Test the regex for commas between XML tags."""
    # Within-chunk commas
    assert HYBRID_COMMA_RE.sub('/>\n<', '<tag a="1"/>,\n  <tag b="2"/>') == '<tag a="1"/>\n<tag b="2"/>'
    assert HYBRID_COMMA_RE.sub('/>\n<', '<tag/>,<tag2/>') == '<tag/>\n<tag2/>'
    assert HYBRID_COMMA_RE.sub('/>\n<', '<tag/>, <tag2/>') == '<tag/>\n<tag2/>'
    assert HYBRID_COMMA_RE.sub('/>\n<', '<tag/>  ,  <tag2/>') == '<tag/>\n<tag2/>'
    # No comma - should not change
    assert HYBRID_COMMA_RE.sub('/>\n<', '<tag/> <tag2/>') == '<tag/> <tag2/>'
    print("PASS: hybrid comma regex")


def test_trailing_comma_regex():
    """Test regex for trailing commas after />."""
    assert TRAILING_COMMA_RE.sub(r'\1', '<tag/>,') == '<tag/>'
    assert TRAILING_COMMA_RE.sub(r'\1', '<tag/>,\n') == '<tag/>'
    assert TRAILING_COMMA_RE.sub(r'\1', '<tag/>,  ') == '<tag/>'
    # No trailing comma
    assert TRAILING_COMMA_RE.sub(r'\1', '<tag/>') == '<tag/>'
    print("PASS: trailing comma regex")


def test_full_adapter_flow():
    """Test the full flow: hybrid format chunks -> adapter -> JSON commands."""
    results = []
    
    def on_partial(name, props):
        results.append(('partial', name, dict(props)))
    
    def on_cmd(name, props):
        results.append(('cmd', name, dict(props)))
    
    adapter = XmlToolStreamAdapter(
        partial_cmd=on_partial,
        cmd=on_cmd,
        speak_command_name='speak',
        emit_partial_on_chars=1,
    )
    
    # Simulate what the pipe would feed after stripping brackets and commas
    cleaned_chunks = [
        '<tell_and_continue text="Hello! This is a test of the XML format for tell_and_continue."/>\n',
        '<tell_and_continue text="Second message to confirm it works with multiple calls."/>\n',
        '<wait_for_user_reply text="Did both messages come through?"/>'
    ]
    
    for chunk in cleaned_chunks:
        adapter.feed(chunk)
    adapter.finish()
    
    print("\nAdapter results:")
    for r in results:
        print(f"  {r}")
    
    cmds = [r for r in results if r[0] == 'cmd']
    assert len(cmds) == 3, f"Expected 3 commands, got {len(cmds)}"
    
    assert cmds[0][1] == 'tell_and_continue'
    assert cmds[0][2]['text'] == 'Hello! This is a test of the XML format for tell_and_continue.'
    
    assert cmds[1][1] == 'tell_and_continue'
    assert cmds[1][2]['text'] == 'Second message to confirm it works with multiple calls.'
    
    assert cmds[2][1] == 'wait_for_user_reply'
    assert cmds[2][2]['text'] == 'Did both messages come through?'
    
    print("PASS: full adapter flow")


def test_chunk_cleaning_pipeline():
    """Test the full chunk cleaning pipeline as it would work in the pipe."""
    # Simulate the streaming chunks from the user's failing case
    # Original LLM output:
    # [
    #   <tell_and_continue text="Hello!"/>,
    #   <tell_and_continue text="Second."/>,
    #   <wait_for_user_reply text="OK?"/>
    # ]
    
    # After first chunk detection and [ stripping:
    raw_chunks = [
        '  <tell_and_continue text="Hello! This is a test of the XML format for tell_and_continue."/>,\n',
        '  <tell_and_continue text="Second message to confirm it works with multiple calls."/>,\n',
        '  <wait_for_user_reply text="Did both messages come through?"/>\n',
    ]
    
    results = []
    
    def on_partial(name, props):
        results.append(('partial', name, dict(props)))
    
    def on_cmd(name, props):
        results.append(('cmd', name, dict(props)))
    
    adapter = XmlToolStreamAdapter(
        partial_cmd=on_partial,
        cmd=on_cmd,
        speak_command_name='speak',
        emit_partial_on_chars=1,
    )
    
    prev_chunk_tail = ''
    
    for i, chunk in enumerate(raw_chunks):
        is_finish = (i == len(raw_chunks) - 1)
        
        # Cross-chunk comma: if prev ended with > (or />,) and this starts with ,
        if prev_chunk_tail.rstrip().endswith('>') and chunk.lstrip().startswith(','):
            chunk = chunk.lstrip()
            if chunk.startswith(','):
                chunk = chunk[1:].lstrip('\n ')
        
        # Also check if prev ended with comma (after />)
        if prev_chunk_tail.rstrip().endswith(',') and chunk.lstrip().startswith('<'):
            # The comma was a separator - we should have stripped it from prev
            # But since we already fed prev, we just skip whitespace at start of this chunk
            chunk = chunk.lstrip('\n ')
        
        # Within-chunk comma replacement
        chunk = HYBRID_COMMA_RE.sub('/>\n<', chunk)
        
        # Strip trailing commas after />
        chunk = TRAILING_COMMA_RE.sub(r'\1', chunk)
        
        # Strip trailing ] on finish
        if is_finish:
            chunk = chunk.rstrip()
            if chunk.endswith(']'):
                chunk = chunk[:-1].rstrip()

        # Strip trailing ] on finish
        if is_finish:
            chunk = chunk.rstrip()
            if chunk.endswith(']'):
                chunk = chunk[:-1].rstrip()
        
        # Save tail for next chunk
        prev_chunk_tail = chunk[-20:] if len(chunk) > 20 else chunk
        
        print(f"Chunk {i}: '{chunk}'")
        
        # Always feed the chunk, then finish on last
        adapter.feed(chunk)
        if is_finish:
            adapter.finish()    
    print("\nAdapter results:")
    for r in results:
        print(f"  {r}")
    
    cmds = [r for r in results if r[0] == 'cmd']
    assert len(cmds) == 3, f"Expected 3 commands, got {len(cmds)}: {cmds}"
    
    assert cmds[0][1] == 'tell_and_continue'
    assert cmds[0][2]['text'] == 'Hello! This is a test of the XML format for tell_and_continue.'
    
    assert cmds[1][1] == 'tell_and_continue'
    assert cmds[1][2]['text'] == 'Second message to confirm it works with multiple calls.'
    
    assert cmds[2][1] == 'wait_for_user_reply'
    assert cmds[2][2]['text'] == 'Did both messages come through?'
    
    print("PASS: chunk cleaning pipeline")


def test_speech_mixed_with_tags():
    """Test that speech text between tags is handled correctly."""
    results = []
    
    def on_partial(name, props):
        results.append(('partial', name, dict(props)))
    
    def on_cmd(name, props):
        results.append(('cmd', name, dict(props)))
    
    adapter = XmlToolStreamAdapter(
        partial_cmd=on_partial,
        cmd=on_cmd,
        speak_command_name='speak',
        emit_partial_on_chars=1,
    )
    
    # Speech before a tag
    adapter.feed('Hello there. ')
    adapter.feed('<tell_and_continue text="I will help."/>')
    adapter.finish()
    
    print("\nSpeech + tag results:")
    for r in results:
        print(f"  {r}")
    
    # Should have a speak partial and a tell_and_continue command
    partials = [r for r in results if r[0] == 'partial']
    cmds = [r for r in results if r[0] == 'cmd']
    
    assert len(cmds) >= 1, f"Expected at least 1 command"
    assert cmds[0][1] == 'tell_and_continue'
    print("PASS: speech mixed with tags")


if __name__ == '__main__':
    test_hybrid_comma_regex()
    test_trailing_comma_regex()
    test_full_adapter_flow()
    test_chunk_cleaning_pipeline()
    test_speech_mixed_with_tags()
    print("\nAll tests passed!")
