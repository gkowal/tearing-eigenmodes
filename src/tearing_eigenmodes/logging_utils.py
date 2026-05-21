import logging
import sys
from typing import Optional

class SmartStreamHandler(logging.StreamHandler):
    """A logging handler that doesn't add a newline if the message starts with \r."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if msg.startswith('\r'):
                self.terminator = ''
            else:
                self.terminator = '\n'
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure the logging system for the project.

    - Console handler uses SmartStreamHandler with INFO (or DEBUG if verbose).
    - File handler (if provided) always uses DEBUG level.
    - Timestamps are included in the file log, but console remains clean.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Allow all levels at root, handlers will filter

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler: Always clean (no timestamps/metadata)
    console_handler = SmartStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    # File handler: Detailed (includes timestamps and metadata)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)
