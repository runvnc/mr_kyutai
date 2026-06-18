"""
Backup of the original process_system_message pipe.
Not imported anywhere - kept for reference.
"""

import json
import os
from typing import Any, Dict
import re

from lib.pipelines.pipe import pipe
from lib.xml_docstring_adapter import convert_docstring_json_examples_to_xml, convert_system_message_for_xml

# Cache: maps original system message text (minus datetime) to converted text.
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
    if not text:
        return data

    global _sysmsg_cache
    cache_key = _strip_datetime(text)
    cached = _sysmsg_cache.get(cache_key)
    if cached is not None:
        return {'text': cached}

    converted = convert_system_message_for_xml(text)
    print("proc sys msg: converting doc string examples:",len(text))
    print(f"xml SYSMSG: converted system message ({len(text)} -> {len(converted)} chars)")
    # Show a snippet of the conversion
    if converted != text:
        # Find first difference
        for i, (a, b) in enumerate(zip(text, converted)):
            if a != b:
                print(f"xml SYSMSG: first diff at char {i}: original={repr(text[max(0,i-10):i+30])} -> converted={repr(converted[max(0,i-10):i+30])}")
                break
        else:
            if len(text) != len(converted):
                print(f"xml SYSMSG: texts differ in length ({len(text)} vs {len(converted)})")
    
    # Cache the result
    _sysmsg_cache[cache_key] = converted
    if len(_sysmsg_cache) > _SYSMSG_CACHE_MAX:
        keys = list(_sysmsg_cache.keys())
        for k in keys[:_SYSMSG_CACHE_MAX // 2]:
           del _sysmsg_cache[k]

    return {'text': converted}
