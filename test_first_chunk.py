#!/usr/bin/env python3
"""Test that hybrid format detection works when [ and <tag are in different chunks."""

import sys
sys.path.insert(0, '/files/mindroot/src')

# Simulate the ACTUAL streaming chunks from the log
chunks = [
    '[\n',
    '  <tell_and_continue text="Hello!"/>,\n',
    '  <wait_for_user_reply text="OK?"/>\n',
    ']'
]

# Current detection logic: first chunk is '[\n'
first = chunks[0]
stripped = first.lstrip()
print(f'First chunk repr: {repr(first)}')
print(f'Stripped: {repr(stripped)}')
print(f'Starts with [: {stripped.startswith("[")}')
after_bracket = stripped[1:].lstrip()
print(f'After bracket: {repr(after_bracket)}')
print(f'Starts with <: {after_bracket.startswith("<")}')
print()
print('BUG: after_bracket is empty, so we fall through to json mode!')
print()
print('Fix: when first chunk is just [ with whitespace, defer detection')
print('     until we have a chunk with actual content after the bracket')
