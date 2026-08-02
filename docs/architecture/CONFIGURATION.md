# Configuration Architecture

This document explains the purpose of the `config` package in **Daedalus** and why the project separates configuration into `settings.py` and `constants.py`.

---

# Configuration Philosophy

A good software project separates **application design** from **deployment configuration**.

Not everything should be configurable, and not everything should be hardcoded.

The guiding principle is simple:

- **`constants.py`** → Things that are part of the application's design and never change.
- **`settings.py`** → Things that may change depending on the machine, deployment, or environment.

---

# `constants.py`

`constants.py` acts as the **blueprint** of the project.

It contains values that define how Daedalus is designed to work.

These values are **not affected** by:

- Operating system
- Laptop
- Server
- Cloud deployment
- Environment variables

## Examples

### Supported File Types

```python
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".ipynb",
}
```

Should this depend on the user's laptop?

**No.**

Daedalus supports these file formats by design.

---

### Default Chunk Size

```python
DEFAULT_CHUNK_SIZE = 1000
```

This is not a machine configuration.

It represents how the application's chunking pipeline was designed.

---

### Application Version

```python
VERSION = "0.1.0"
```

The version is project metadata.

It is **not** an environment variable.

---

## Typical Constants

Examples of values that belong in `constants.py`:

```python
APP_NAME
VERSION
SUPPORTED_EXTENSIONS
DEFAULT_TOP_K
DEFAULT_CHUNK_SIZE
RRF_K
```

These values remain consistent regardless of where Daedalus runs.

---

# `settings.py`

`settings.py` is completely different.

It contains values that **may change between environments**.

These values are loaded from environment variables (`.env`) using **Pydantic Settings**.

## Example

Today you use:

```text
qwen3:8b
```

Tomorrow you upgrade your GPU.

Now you want:

```text
qwen3:30b
```

Should you edit Python files?

**No.**

Simply change your `.env` file:

```env
DAEDALUS_LLM_MODEL=qwen3:30b
```

No code changes are required.

---

## Another Example

Local development:

```text
http://localhost:11434
```

Cloud deployment:

```text
http://10.10.1.15:11434
```

Again, only the `.env` file changes.

Your Python code remains exactly the same.

---

# Why Use `BaseSettings`?

Without `BaseSettings`, every module would need to manually read environment variables.

Example:

```python
import os

MODEL = os.environ.get("DAEDALUS_LLM_MODEL")
```

Imagine repeating this in:

- `retrieval.py`
- `interview.py`
- `generation.py`
- `worker.py`
- `evaluation.py`
- `api.py`
- `storage.py`

That creates unnecessary duplication.

Instead, everything is centralized into a single object:

```python
settings
```

Every module imports the same configuration object.

---

# What Happens Internally?

When Python starts, this line executes:

```python
settings = Settings()
```

Internally, Pydantic performs the following steps:

```text
Read .env
      │
      ▼
Read Environment Variables
      │
      ▼
Validate Data Types
      │
      ▼
Create Settings Object
      │
      ▼
Ready to Use
```

Once created, the `settings` object stays in memory for the lifetime of the application.

---

# Example

Suppose your `.env` contains:

```env
DAEDALUS_LLM_MODEL=qwen3:14b
```

When this runs:

```python
settings = Settings()
```

It becomes:

```python
settings.llm_model
```

From that point onward, your application never manually reads `.env` again.

---

# How Modules Use `settings`

Suppose you later create a module:

```text
generation/
└── generator.py
```

Inside the module:

```python
from daedalus.config import settings
```

Now you can simply access:

```python
settings.llm_model
```

without worrying about where the value came from.

---

## Retrieval Module

```python
from daedalus.config import settings
```

Uses:

```python
settings.embedding_model
```

---

## Interview Engine

```python
from daedalus.config import settings
```

Uses:

```python
settings.routing_model
```

---

## Vision Pipeline

```python
from daedalus.config import settings
```

Uses:

```python
settings.vision_model
```

Every module shares the **same configuration object**.

There is:

- No duplication
- No repeated environment lookups
- No inconsistent configuration

---

# How Modules Use `constants.py`

Unlike `settings.py`, constants are imported directly.

## Example: Upload Validation

```python
from daedalus.config.constants import SUPPORTED_EXTENSIONS

if suffix not in SUPPORTED_EXTENSIONS:
    raise ValueError("Unsupported file type.")
```

---

## Example: Chunking

```python
from daedalus.config.constants import DEFAULT_CHUNK_SIZE

chunk_size = DEFAULT_CHUNK_SIZE
```

---

## Example: Retrieval

```python
from daedalus.config.constants import DEFAULT_TOP_K

top_k = DEFAULT_TOP_K
```

No environment variables.

No `.env`.

Because these values never change.

---

# Overall Architecture

```text
                    .env
                      │
                      │
             pydantic-settings
                      │
                      ▼
                settings.py
                      │
          settings = Settings()
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   Generation     Retrieval     Interview
        │             │             │
        ▼             ▼             ▼
 settings.llm   settings.embed  settings.routing
```

---

# Constants Architecture

```text
               constants.py
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
 Ingestion      Retrieval      Chunking
      │              │              │
      ▼              ▼              ▼
SUPPORTED_      DEFAULT_TOP_K  DEFAULT_CHUNK_SIZE
EXTENSIONS
```

Any module can directly import the constant it needs.

---

# Why Not Put Everything in `settings.py`?

Because not everything should be configurable.

Consider:

```python
settings.default_chunk_size
```

This suggests that users should modify the chunk size.

But should they?

Usually **no**.

Chunk size is a design decision made by the developers.

Therefore it belongs in:

```text
constants.py
```

---

# Why Not Put Everything in `constants.py`?

Because deployment configuration changes frequently.

Imagine changing:

```text
qwen3:8b
```

to

```text
qwen3:30b
```

If models were stored in `constants.py`, every deployment would require editing Python files.

That means:

- Every laptop
- Every server
- Every cloud deployment

would require code changes.

This defeats the purpose of configuration management.

---

# The Philosophy

A simple rule that scales well is:

### `settings.py`

> **"What might change between environments?"**

Examples:

- Models
- Embedding models
- URLs
- Ports
- Debug flags
- Credentials
- Service endpoints
- Configurable paths

---

### `constants.py`

> **"What is part of Daedalus' design regardless of where it runs?"**

Examples:

- Supported file formats
- Default algorithms
- Fixed limits
- Application metadata
- Default retrieval values
- Chunking defaults

---

# Final Takeaway

As Daedalus grows with modules such as:

- Ingestion
- Retrieval
- Generation
- Interview Engine
- Evaluation
- Storage

almost every module will import either:

```python
from daedalus.config import settings
```

or

```python
from daedalus.config.constants import SOME_CONSTANT
```

This clear separation keeps the codebase:

- Easy to understand
- Easy to maintain
- Easy to deploy
- Easy to scale

Most importantly, it prevents deployment-specific configuration from being mixed with the application's core design.
