print("Loading Kyutai")

from lib.providers.commands import command

@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using Kyutai streaming."""
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id
    sip_response_started = False

    try:
        print(f"speak() CALLED text='{text[:60]}...' log_id={log_id}")

print("OK")
