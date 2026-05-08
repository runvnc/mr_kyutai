from typing import Optional, Dict, Any

print("Loading Kyutai")

from mindroot.lib.providers.commands import command

@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using Kyutai streaming TTS.

    In realtime mode (MR_KYUTAI_REALTIME_STREAM=1), the actual audio
    generation is handled by the partial_command pipe which intercepts
    incremental speak() calls as the LLM streams. This command exists
    so it can be assigned to agents.
    """
    pass

print("OK")
