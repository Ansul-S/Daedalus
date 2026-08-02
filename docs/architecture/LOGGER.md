###Theory

# Logging in Daedalus

---

# Why Logging Exists

Imagine **Daedalus** without logging.

A user uploads a PDF.

Internally, the application performs the following steps:

```text
Upload
   │
   ▼
Extract PDF
   │
   ▼
Chunk Document
   │
   ▼
Generate Embeddings
   │
   ▼
Store Vectors
   │
   ▼
Done
```

Everything works perfectly.

Great.

---

Now imagine a user reports:

> "My upload failed."

Without logging, all you know is:

```text
500 Internal Server Error
```

That tells you **nothing** about what actually failed.

Was it:

- PDF extraction?
- Chunking?
- Embedding generation?
- Database storage?
- Upload itself?

There is no way to know.

---

# How Logging Solves This

With logging enabled, the application records every important event.

```text
INFO     Upload started
INFO     Reading PDF
INFO     PDF parsed
INFO     Creating chunks
INFO     Generated 52 chunks
INFO     Embeddings started
ERROR    Ollama unavailable
Traceback...
```

Immediately you know:

✅ Upload succeeded

✅ PDF parsing succeeded

✅ Chunking succeeded

❌ Embedding generation failed

Instead of searching through the entire pipeline, you know the exact stage where the failure occurred.

---

# Think of Logging as a Diary

A simple way to understand logging is to imagine it as a diary that your application writes while it is running.

Example:

```text
13:00  Server started
13:02  User uploaded notes.pdf
13:02  PDF parsed
13:02  Generated 52 chunks
13:03  Embeddings completed
13:03  Saved to database
```

Every important event gets written down.

Later, if something goes wrong, you simply read the diary.

---

# Why Not Just Use `print()`?

Many beginners start with:

```python
print("Upload started")
```

This works during development because you're watching the terminal.

However, after deployment:

- the application may run on another server
- inside Docker
- on AWS
- on Railway
- inside Kubernetes
- in the cloud

Nobody is watching your terminal output.

Production applications rely on logging systems that automatically collect, organize, and preserve logs.

Instead of using `print()`, production code uses:

```python
logger.info("Upload started")
```

---

# Python Already Includes Logging

Python has a built-in logging library.

No external package is required.

```python
import logging
```

That's all you need to start.

---

# Logging Levels

Not every message has the same importance.

Python classifies log messages into different severity levels.

---

## DEBUG

Very detailed information intended for developers.

Example:

```text
Chunk size = 1000
Overlap = 200
Current model = qwen3:8b
```

Typical use cases:

- Variable values
- Internal calculations
- Performance measurements
- Function entry/exit

Normally hidden in production.

---

## INFO

Normal application events.

Examples:

```text
Application started
Upload received
Document parsed
Interview started
Embeddings generated
```

This is the most commonly used logging level.

---

## WARNING

Something unexpected happened, but the application can continue.

Examples:

```text
Large PDF uploaded
Model fallback used
Document contains no text
```

The application still works, but attention may be required.

---

## ERROR

Something failed.

Examples:

```text
Embedding generation failed
Cannot connect to Ollama
SQLite database locked
```

An operation could not be completed.

---

## CRITICAL

A fatal error occurred.

The application cannot continue.

Examples:

```text
Database corrupted
Configuration missing
Cannot start server
```

Usually followed by application shutdown.

---

# Real Daedalus Example

Suppose a user uploads:

```text
DeepLearning.pdf
```

Successful execution produces logs similar to:

```text
INFO  Upload received
INFO  Saved to uploads/
INFO  Starting ingestion
INFO  Reading PDF
INFO  45 pages detected
INFO  Generated 181 chunks
INFO  Creating embeddings
INFO  Stored vectors
INFO  Completed
```

Everything completed successfully.

---

Now imagine Ollama is unavailable.

The logs become:

```text
INFO  Reading PDF
INFO  Generated 181 chunks
INFO  Embedding started
ERROR Cannot connect to Ollama

Traceback...
Connection refused
```

The failure location becomes immediately obvious.

---

# Another Example: Interview Engine

Suppose the interview workflow looks like this:

```text
User Answers Question
          │
          ▼
LLM Evaluates Answer
          │
          ▼
Generate Score
```

Corresponding logs:

```text
INFO  Interview started
INFO  Question #3
INFO  Evaluation started
INFO  Score = 8.2
INFO  Interview finished
```

Every important event is recorded.

---

# Logging Helps Even Months Later

Imagine someone reports:

> "Yesterday my upload disappeared."

Without logs:

```text
Impossible to investigate.
```

With logs:

```text
14:05 Upload received
14:05 Storage full
14:05 Upload aborted
```

The problem becomes immediately clear.

---

# Where Does `logger` Come From?

During **Phase 3**, we'll create a central logging configuration.

Project structure:

```text
daedalus/
│
├── core/
│   └── logging.py
│
├── ingestion/
├── retrieval/
├── evaluation/
└── ...
```

The `core/logging.py` module configures logging once for the entire application.

Every other module simply uses that configuration.

---

# Using a Logger Inside a Module

Example:

```text
ingestion/pdf.py
```

At the top of the file:

```python
import logging

logger = logging.getLogger(__name__)
```

Now you can write:

```python
logger.info("Reading PDF")
```

Or:

```python
logger.info("Generating chunks")
logger.info("Creating embeddings")
logger.warning("PDF contains no extractable text")
logger.error("Embedding generation failed")
```

The module only reports events.

It does **not** configure logging itself.

---

# What Does `__name__` Mean?

Suppose the current file is:

```text
daedalus/ingestion/pdf.py
```

Then:

```python
__name__
```

becomes:

```text
daedalus.ingestion.pdf
```

So your logs automatically include the module name.

Example:

```text
INFO  daedalus.ingestion.pdf
      Reading PDF
```

Another module:

```text
daedalus/retrieval/search.py
```

Produces:

```text
INFO  daedalus.retrieval.search
      Searching vectors
```

This makes it easy to identify exactly which part of the application generated each message.

---

# How Logging Flows Through the Application

```text
logger.info(...)
        │
        ▼
Python Logging Module
        │
        ▼
Formatter
        │
        ▼
Console Output
        │
        ├────────► Log File (optional)
        │
        └────────► Cloud Logging (optional)
```

Every module sends messages into the same centralized logging system.

---

# What We'll Build in Phase 3

We'll create:

```text
core/
└── logging.py
```

This module will:

- Configure the application's logging format.
- Set the default log level (typically `INFO` during development).
- Decide where logs are written (console initially, files later if desired).
- Ensure every part of Daedalus produces consistent log output.

Then every module only needs:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting document ingestion")
logger.warning("Document contains no extractable text")
logger.error("Embedding generation failed")
```

Notice that the module never worries about formatting, timestamps, or output destinations.

Those responsibilities belong entirely to the centralized configuration in `core/logging.py`.

---

# The Right Mindset

A useful way to think about logging is:

| `print()` | Logging |
|-----------|---------|
| Temporary debugging | Permanent application history |
| Used while writing code | Used while running software |
| Visible only in your terminal | Collected by logging systems |
| Usually removed later | Remains part of production code |

---

# Key Takeaways

For a project like **Daedalus**, logging is essential because the application contains multiple independent stages:

- Document uploads
- PDF parsing
- OCR
- Text chunking
- Embedding generation
- Vector storage
- Retrieval
- LLM inference
- Interview evaluation

Each stage can fail independently.

Without logging, diagnosing those failures becomes slow and frustrating.

With logging, every important event is recorded in chronological order, making debugging, monitoring, and long-term maintenance significantly easier.

As Daedalus grows, logging will become one of the most valuable tools for understanding what the application did—not just while you're developing it, but weeks or even months after it has been deployed.


###Code

# Logging System Documentation

## Introduction

Logging is the process of recording events that occur while an application is running.

Unlike `print()` statements, which simply display text to the console, a logging system provides a structured way to record information about an application's execution.

Logs help developers:

- Monitor application behavior
- Debug unexpected errors
- Trace execution flow
- Measure performance
- Record important events
- Diagnose production failures

For an AI application like **Daedalus**, logging will eventually record events such as:

- API requests
- Document uploads
- PDF parsing
- Embedding generation
- Retrieval operations
- LLM requests
- Evaluation results
- System errors

Without logging, diagnosing failures in a large application becomes extremely difficult.

---

# Why Not Use `print()`?

Many beginners start with simple statements such as:

```python
print("Application started")
```

Although this works for small scripts, it has several limitations.

`print()`:

- Has no severity levels
- Cannot be filtered
- Has inconsistent formatting
- Cannot write to log files automatically
- Cannot be redirected easily
- Provides no timestamps
- Cannot identify which module produced the output

Python's built-in `logging` module solves all of these problems and is designed for production applications.

---

# Python Logging Architecture

Understanding the architecture is far more important than memorizing function names.

Internally, Python logging follows this pipeline:

```text
Application
      │
      ▼
Logger
      │
      ▼
LogRecord
      │
      ▼
Handler
      │
      ▼
Formatter
      │
      ▼
Console / File / Database / Cloud
```

Each component has a specific responsibility.

---

# Logger

The **Logger** is responsible for creating log messages.

Example:

```python
logger.info("Application started")
```

The logger **does not print anything itself**.

Instead, it creates a **LogRecord** object containing information such as:

- Timestamp
- Severity level
- Module name
- Filename
- Line number
- Message

This `LogRecord` is then passed to the configured handlers.

---

# Handler

Handlers decide **where** log messages should be sent.

Common handler destinations include:

- Console
- Text file
- Rotating log file
- Email
- Database
- Cloud monitoring systems

In Daedalus we currently use:

```python
logging.StreamHandler(sys.stdout)
```

Meaning:

> Send every log message to the terminal.

Later this can easily expand into:

```text
Console
      +
Rotating Log File
      +
Cloud Logging
```

without changing any application code.

---

# Formatter

Handlers only know **where** to send logs.

The **Formatter** decides **how** they look.

Example output:

```text
2026-07-25 18:32:10 | INFO | daedalus.api.main | Server started
```

Without a formatter, logs would appear in Python's default format, which is less readable.

---

# Root Logger

Python contains one special logger called the **Root Logger**.

Hierarchy:

```text
Root Logger
│
├── daedalus.api
├── daedalus.retrieval
├── daedalus.ingestion
├── daedalus.llm
└── ...
```

Every logger created using

```python
logging.getLogger(__name__)
```

inherits its configuration from the root logger.

This is why logging should be configured **only once** during application startup.

---

# Module Docstring

```python
"""
Central logging configuration for Daedalus.

...
"""
```

Every Python module may begin with a module docstring.

Unlike comments, a docstring becomes part of the module metadata.

A module docstring explains:

- Purpose
- Responsibilities
- Usage

This improves readability and allows documentation generators to include the module automatically.

---

# Future Import

```python
from __future__ import annotations
```

This enables **postponed evaluation of type annotations**.

Normally:

```python
def f(x: MyClass):
```

requires `MyClass` to already exist.

With future annotations enabled, Python stores annotations as strings until they are actually needed.

Benefits include:

- Avoids circular imports
- Improves compatibility
- Slightly speeds startup
- Recommended for modern Python projects

---

# Importing Modules

## Importing the Logging Library

```python
import logging
```

Imports Python's standard logging library.

This module provides everything needed for production logging:

- Logger
- Handler
- Formatter
- Filter
- LogRecord
- Log Levels

---

## Importing `sys`

```python
import sys
```

Provides access to Python runtime objects.

We specifically use:

```python
sys.stdout
```

Meaning:

> Write log messages to the terminal output stream.

---

## Importing Project Settings

```python
from daedalus.config import settings
```

Imports the centralized runtime configuration created during **Phase 2**.

Instead of hardcoding:

```python
logging.INFO
```

the application reads:

```python
settings.log_level
```

This follows Daedalus' centralized configuration architecture.

Changing:

```text
INFO
```

to

```text
DEBUG
```

requires **no code changes**.

---

# Public API

```python
__all__ = ["configure_logging"]
```

Every Python module has a public interface.

`__all__` explicitly defines which objects belong to that interface.

Meaning:

```python
from daedalus.core.logging import *
```

imports only:

```python
configure_logging
```

Everything else remains an implementation detail.

Although primarily enforced by convention, this improves encapsulation and clearly communicates the intended public API.

---

# Logging Format

```python
LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)
```

This defines how every log entry is displayed.

The formatter uses placeholder variables.

---

## `%(asctime)s`

Displays the timestamp.

Example:

```text
2026-07-25 18:30:11
```

---

## `%(levelname)s`

Displays the severity level.

Examples:

```text
INFO
WARNING
ERROR
```

---

## `-8s`

This is standard Python string formatting.

Meaning:

- Left align the text
- Width = 8 characters

Without alignment:

```text
INFO
WARNING
ERROR
```

The log columns become uneven.

With `-8s`:

```text
INFO
WARNING
ERROR
```

Shorter names such as `INFO` are padded with spaces so all log columns align neatly.

---

## `%(name)s`

Displays the logger name.

Usually this is the module path.

Example:

```text
daedalus.retrieval.search
```

This immediately identifies which module generated the message.

---

## `%(message)s`

Displays the message supplied by the developer.

Example:

```python
logger.info("Upload complete")
```

Produces:

```text
Upload complete
```

---

# Date Format

```python
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
```

Controls timestamp formatting.

Example output:

```text
2026-07-25 19:06:42
```

instead of Python's default timestamp representation.

---

# `configure_logging()`

This function configures the application's entire logging system.

It should be called **exactly once** during application startup.

Every other module simply obtains a logger.

This follows the principle of:

> **Single initialization, many consumers.**

---

# Reading the Log Level

```python
log_level = getattr(
    logging,
    settings.log_level.upper(),
    logging.INFO,
)
```

This dynamically converts the configured log level into a logging constant.

Suppose:

```python
settings.log_level = "debug"
```

First:

```python
.upper()
```

converts it into:

```text
DEBUG
```

Then:

```python
getattr(logging, "DEBUG")
```

returns:

```python
logging.DEBUG
```

If an invalid value is supplied:

```text
VERBOSE
```

then:

```python
logging.INFO
```

is used as a safe fallback.

This prevents runtime crashes caused by configuration mistakes.

---

# Handlers

```python
handlers = [
    logging.StreamHandler(sys.stdout),
]
```

A list is used even though there is currently only one handler.

Why?

Because the system is designed to scale.

Later the configuration can become:

```python
handlers = [
    logging.StreamHandler(sys.stdout),
    RotatingFileHandler(...),
]
```

No other application code needs to change.

---

# `logging.basicConfig()`

This is Python's central logging configuration function.

It initializes:

- Root logger
- Formatter
- Handlers
- Log level

All at once.

Every logger created afterward automatically inherits this configuration.

---

# `force=True`

```python
force=True
```

This tells Python:

> Remove any existing logging configuration before applying this one.

This is especially useful with **Uvicorn**, which may configure logging before your application starts.

Without `force=True`, your configuration may be ignored or mixed with existing handlers, resulting in duplicate or inconsistent log output.

---

# Third-Party Loggers

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
```

Some libraries produce a large amount of informational output.

For example:

- Every HTTP request
- Every connection
- Every retry

These logs are useful while developing the library itself but usually clutter application logs.

Raising the logging level to `WARNING` suppresses routine informational messages while still displaying warnings and errors.

The same technique can be applied to libraries such as:

- `urllib3`
- `asyncio`

---

# Startup Log

```python
logging.getLogger().info("Logging initialized.")
```

This writes a message using the root logger.

Purpose:

- Confirms logging has been configured successfully
- Creates a clear startup event in the logs

Expected output:

```text
2026-07-25 19:06:42 | INFO | root | Logging initialized.
```

---

# How Other Modules Use Logging

Every other module follows the same pattern.

```python
import logging

logger = logging.getLogger(__name__)
```

Examples:

```python
logger.info("Upload started")

logger.warning("Unsupported file type")

logger.error("Embedding generation failed")
```

No module should call:

- `configure_logging()`
- `logging.basicConfig()`

again.

Instead, every module simply obtains a logger and emits messages.

This keeps the logging configuration centralized and consistent across the entire application.

---

# Design Principles Demonstrated

This logging module reflects several important software engineering principles.

## Centralized Configuration

Logging is configured in one place only.

---

## Single Responsibility Principle (SRP)

The module's only responsibility is configuring the logging system.

---

## Separation of Concerns

Configuration is separated from log generation.

Modules emit log messages but do not decide:

- How logs are formatted
- Where logs are written
- Which handlers are used

---

## Configuration Over Hardcoding

The logging level is obtained from `settings.py`.

This allows behavior to change without modifying application code.

---

## Extensibility

The handler list is intentionally designed for future expansion.

Possible future additions include:

- Rotating file handlers
- JSON logging
- Cloud logging services
- Database logging

without requiring changes throughout the application.

---

## Consistency

Every module inherits:

- The same formatter
- The same handlers
- The same log levels
- The same configuration

from the Root Logger.

---

# Summary

The Daedalus logging system is designed around centralized configuration and modularity.

Only one module is responsible for configuring logging, while every other component simply requests a logger and emits messages.

This architecture provides:

- Consistent log formatting
- Centralized configuration
- Easy debugging
- Scalability
- Maintainability
- Extensibility

As Daedalus grows to include document ingestion, retrieval, LLM pipelines, evaluation systems, and API services, every component will share the same logging infrastructure while remaining completely independent of its implementation details.
