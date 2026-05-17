import logging
import sys

class SmartStreamHandler(logging.StreamHandler):
    """A logging handler that doesn't add a newline if the message starts with \r."""
    def emit(self, record):
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

def setup_logging(verbose=False, log_file=None):
    """
    Configure the logging system for the project.
    
    - Console handler uses SmartStreamHandler with INFO (or DEBUG if verbose).
    - File handler (if provided) always uses DEBUG level.
    - Timestamps are included in both.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Define formats
    console_fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    # If using SmartStreamHandler for interactive progress, we might want a simpler format 
    # for progress lines specifically, but Formatter applies to all.
    # Let's use a simpler format for console to keep it clean.
    console_fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Allow all levels at root, handlers will filter
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Console handler
    console_handler = SmartStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)
