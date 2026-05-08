# MindRoot plugin package
# When running as a standalone server (e.g. python -m mr_kyutai.remote_server),
# MindRoot is not available, so we guard these imports.
try:
    from .mod import *
    from . import realtime_stream  # register partial_command pipe
except ImportError:
    # Running standalone (remote_server mode) - MindRoot not available
    import logging
    logging.getLogger(__name__).debug("MindRoot not available, skipping plugin imports")
