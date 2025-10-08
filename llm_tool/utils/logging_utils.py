#!/usr/bin/env python3
"""
PROJECT:
-------
LLMTool

TITLE:
------
logging_utils.py

MAIN OBJECTIVE:
---------------
This script provides comprehensive logging utilities for the LLMTool package
including structured logging, log rotation, and performance tracking.

Dependencies:
-------------
- sys
- logging
- pathlib
- typing
- json
- datetime

MAIN FEATURES:
--------------
1) Structured JSON logging
2) Log rotation and archiving
3) Performance metrics logging
4) Error tracking and aggregation
5) Progress logging with context

Author:
-------
Antoine Lemor
"""

import logging
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextlib import contextmanager
import traceback

# Try to import Rich for enhanced console output
try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    RichHandler = None
    Console = None


class StructuredLogger:
    """Structured logger with JSON output support"""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        level: str = 'INFO',
        use_console: bool = True,
        use_file: bool = True,
        use_json: bool = False
    ):
        """
        Initialize structured logger.
        
        Parameters
        ----------
        name : str
            Logger name
        log_dir : Path, optional
            Directory for log files
        level : str
            Logging level
        use_console : bool
            Whether to log to console
        use_file : bool
            Whether to log to file
        use_json : bool
            Whether to use JSON format
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Setup log directory
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use application subdirectory for general logs
            self.log_dir = Path('logs/application')
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_json = use_json
        
        # Setup handlers
        if use_console:
            self._setup_console_handler()
        if use_file:
            self._setup_file_handler()

    def _setup_console_handler(self):
        """Setup console logging handler"""
        if HAS_RICH and not self.use_json:
            # Use Rich handler for better console output
            console_handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_level=True
            )
            console_handler.setFormatter(
                logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            )
        else:
            # Standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            if self.use_json:
                console_handler.setFormatter(JsonFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '[%(asctime)s] [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                )
        
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self):
        """Setup file logging handler with rotation"""
        # Main log file
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        if self.use_json:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        self.logger.addHandler(file_handler)
        
        # Error log file (separate for errors and above)
        error_log = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s\n%(exc_info)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        self.logger.addHandler(error_handler)

    def log(self, level: str, message: str, **context):
        """Log with additional context"""
        if context:
            if self.use_json:
                # Include context in message for JSON
                extra = {'context': context}
                getattr(self.logger, level.lower())(message, extra=extra)
            else:
                # Append context to message
                context_str = ' | '.join(f"{k}={v}" for k, v in context.items())
                full_message = f"{message} | {context_str}"
                getattr(self.logger, level.lower())(full_message)
        else:
            getattr(self.logger, level.lower())(message)

    def info(self, message: str, **context):
        """Log info message"""
        self.log('info', message, **context)

    def warning(self, message: str, **context):
        """Log warning message"""
        self.log('warning', message, **context)

    def error(self, message: str, exc_info=False, **context):
        """Log error message"""
        if exc_info:
            context['traceback'] = traceback.format_exc()
        self.log('error', message, **context)

    def debug(self, message: str, **context):
        """Log debug message"""
        self.log('debug', message, **context)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra context if available
        if hasattr(record, 'context'):
            log_obj['context'] = record.context
        
        # Add exception info if available
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, ensure_ascii=False)


class PerformanceLogger:
    """Logger for tracking performance metrics"""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """
        Initialize performance logger.
        
        Parameters
        ----------
        logger : StructuredLogger, optional
            Logger to use (creates new if None)
        """
        self.logger = logger or StructuredLogger('performance')
        self.metrics = {}
        self.timers = {}

    @contextmanager
    def timer(self, operation: str, **context):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            
            # Store metric
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(elapsed)
            
            # Log
            self.logger.info(
                f"Operation completed: {operation}",
                duration_seconds=elapsed,
                **context
            )

    def start_timer(self, operation: str):
        """Start a timer for an operation"""
        self.timers[operation] = time.perf_counter()

    def stop_timer(self, operation: str, **context) -> float:
        """Stop a timer and log the duration"""
        if operation not in self.timers:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        elapsed = time.perf_counter() - self.timers[operation]
        del self.timers[operation]
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(elapsed)
        
        # Log
        self.logger.info(
            f"Timer stopped: {operation}",
            duration_seconds=elapsed,
            **context
        )
        
        return elapsed

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if operation:
            if operation not in self.metrics:
                return {}
            
            times = self.metrics[operation]
            return {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        
        # All operations
        stats = {}
        for op, times in self.metrics.items():
            stats[op] = {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        
        return stats


class ErrorAggregator:
    """Aggregate and track errors for analysis"""

    def __init__(self, max_errors: int = 1000):
        """
        Initialize error aggregator.
        
        Parameters
        ----------
        max_errors : int
            Maximum errors to store
        """
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}
        self.logger = logging.getLogger(__name__)

    def add_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ):
        """Add an error to the aggregator"""
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': message,
            'context': context or {}
        }
        
        if exc_info:
            error_entry['traceback'] = traceback.format_exception(
                type(exc_info),
                exc_info,
                exc_info.__traceback__
            )
        
        # Store error
        self.errors.append(error_entry)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
        
        # Count by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'error_types': self.error_counts,
            'recent_errors': self.errors[-10:] if self.errors else []
        }

    def export_errors(self, output_path: Path):
        """Export errors to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'summary': self.get_summary(),
                    'errors': self.errors
                },
                f,
                indent=2,
                ensure_ascii=False
            )
        
        self.logger.info(f"Exported {len(self.errors)} errors to {output_path}")


def setup_logging(
    name: str = 'llm_tool',
    level: str = 'INFO',
    log_dir: Optional[str] = None,
    use_json: bool = False
) -> StructuredLogger:
    """
    Setup logging for the application.
    
    Parameters
    ----------
    name : str
        Logger name
    level : str
        Logging level
    log_dir : str, optional
        Log directory
    use_json : bool
        Whether to use JSON format
    
    Returns
    -------
    StructuredLogger
        Configured logger
    """
    # Suppress verbose third-party loggers
    for logger_name in ['urllib3', 'requests', 'ollama', 'httpx']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Create and return structured logger
    return StructuredLogger(
        name=name,
        log_dir=Path(log_dir) if log_dir else None,
        level=level,
        use_json=use_json
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with console handler"""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Add console handler to show logs in terminal
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)

        # Simple format to not clutter the beautiful colored output
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger
