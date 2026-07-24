# Daedalus `pyproject.toml` Library Guide

This document explains every major library used in the Daedalus project,
what it does, and why it exists in the architecture.

------------------------------------------------------------------------

# Core Backend Dependencies

## FastAPI

**Purpose:** Modern Python web framework for building APIs.

### What it does

FastAPI receives HTTP requests from the frontend and routes them to your
Python code.

Example endpoints:

``` python
@app.post("/upload")
async def upload(file: UploadFile):
    ...
```

### Why Daedalus uses it

-   Upload PDFs and notebooks
-   Start indexing
-   Run interview sessions
-   Answer RAG queries
-   Return citations
-   Future authentication and user management

FastAPI is the **backend brain** of the application.

------------------------------------------------------------------------

## Uvicorn

**Purpose:** ASGI web server.

FastAPI defines your application, but Uvicorn actually runs it.

``` bash
uvicorn app:app
```

Responsibilities:

-   Starts the web server
-   Accepts browser requests
-   Passes them to FastAPI
-   Returns responses

Think of FastAPI as the kitchen and Uvicorn as the waiter.

------------------------------------------------------------------------

## Pydantic

**Purpose:** Data validation and serialization.

Example:

``` python
class Query(BaseModel):
    question: str
    top_k: int
```

Pydantic automatically:

-   validates request data
-   converts compatible types
-   returns clean validation errors

Used throughout Daedalus for:

-   API requests
-   API responses
-   configuration
-   internal models

------------------------------------------------------------------------

## pydantic-settings

Provides typed configuration using environment variables.

Instead of repeatedly calling `os.getenv()`, define:

``` python
class Settings(BaseSettings):
    database_url: str
    jwt_secret: str
```

Benefits:

-   type safety
-   validation
-   defaults
-   cleaner configuration

------------------------------------------------------------------------

## python-dotenv

Loads variables from a local `.env` file.

Example:

``` text
DATABASE_URL=...
SUPABASE_KEY=...
```

Useful during development.

------------------------------------------------------------------------

## python-multipart

Required for browser file uploads.

Without it, FastAPI cannot process:

``` python
UploadFile
```

Used for:

-   PDF uploads
-   Notebook uploads
-   Images

------------------------------------------------------------------------

## aiofiles

Asynchronous file operations.

Instead of blocking the server while writing files:

``` python
async with aiofiles.open(...) as f:
```

Benefits:

-   concurrent uploads
-   faster backend
-   non-blocking I/O

------------------------------------------------------------------------

## httpx

Modern HTTP client supporting async programming.

Used when Daedalus communicates with:

-   Ollama
-   Hugging Face
-   Supabase
-   OpenAI
-   Anthropic
-   other REST APIs

------------------------------------------------------------------------

## orjson

Extremely fast JSON library written in Rust.

Benefits:

-   faster serialization
-   lower latency
-   efficient API responses

------------------------------------------------------------------------

## NumPy

Foundation of scientific computing.

Provides:

-   vectors
-   matrices
-   mathematical operations
-   cosine similarity support

Nearly every ML library depends on NumPy.

------------------------------------------------------------------------

## pymupdf4llm

Extracts structured Markdown from PDFs.

Unlike simple PDF text extraction, it preserves:

-   headings
-   lists
-   tables
-   reading order

Perfect for RAG chunking.

------------------------------------------------------------------------

## nbformat

Reads Jupyter Notebook (`.ipynb`) files.

Allows Daedalus to ingest:

-   lecture notebooks
-   assignments
-   labs
-   tutorials

It exposes notebook cells individually.

------------------------------------------------------------------------

## markdown-it-py

Markdown parser.

Converts Markdown into a structured syntax tree.

Useful for:

-   heading-aware chunking
-   semantic splitting
-   preserving document structure

------------------------------------------------------------------------

# Optional OCR Dependencies

## OCRmyPDF

Adds OCR to scanned PDFs.

Pipeline:

Image PDF → OCR → Searchable PDF

Makes scanned documents usable by the RAG pipeline.

------------------------------------------------------------------------

## PaddleOCR

State-of-the-art OCR for images.

Reads text from:

-   PNG
-   JPG
-   screenshots
-   whiteboards

------------------------------------------------------------------------

## PaddlePaddle

Deep learning framework required by PaddleOCR.

Comparable to:

-   PyTorch
-   TensorFlow

------------------------------------------------------------------------

## OpenCV

Computer vision toolkit.

Typical preprocessing:

-   rotate
-   deskew
-   denoise
-   crop
-   resize

Improves OCR accuracy.

------------------------------------------------------------------------

## Pillow

Python Imaging Library.

Used for:

-   image loading
-   resizing
-   format conversion
-   simple preprocessing

------------------------------------------------------------------------

# Machine Learning Dependencies

## sentence-transformers

Creates embeddings.

Pipeline:

Text → Embedding Vector

Supports models such as:

-   BGE-M3
-   MiniLM
-   E5
-   GTE

Core library for semantic search.

------------------------------------------------------------------------

## transformers

Hugging Face ecosystem.

Loads:

-   LLMs
-   embedding models
-   rerankers
-   tokenizers
-   classifiers

One of the most important AI libraries.

------------------------------------------------------------------------

## sentencepiece

Tokenizer used by many modern language models.

Breaks text into subword tokens for efficient processing.

Used by models like:

-   Llama
-   Gemma
-   T5

------------------------------------------------------------------------

## protobuf

Serialization library.

Stores:

-   model weights
-   configuration
-   metadata

Usually used indirectly by ML frameworks.

------------------------------------------------------------------------

# Development Dependencies

## PyTest

Testing framework.

Automatically discovers and runs tests.

------------------------------------------------------------------------

## pytest-asyncio

Adds async support to PyTest.

Essential for testing FastAPI endpoints.

------------------------------------------------------------------------

## Ruff

Fast linter.

Checks:

-   style
-   formatting
-   common bugs
-   unused imports

------------------------------------------------------------------------

## MyPy

Static type checker.

Detects type errors before runtime.

Improves maintainability.

------------------------------------------------------------------------

## pre-commit

Runs automated checks before every Git commit.

Typical checks:

-   Ruff
-   MyPy
-   Tests

Prevents broken code from entering the repository.

------------------------------------------------------------------------

# Build & Packaging

## Hatchling

Modern Python build backend.

Creates installable packages.

------------------------------------------------------------------------

## src Layout

``` text
src/
    daedalus/
```

Keeps imports clean and avoids accidental local imports.

------------------------------------------------------------------------

# Testing Configuration

PyTest searches only inside:

``` text
tests/
```

------------------------------------------------------------------------

# Ruff Configuration

-   Python target: 3.11
-   Maximum line length: 100
-   Source directory: `src`

------------------------------------------------------------------------

# Overall Architecture

``` text
                   USER
                     │
                     ▼
            Next.js Frontend
                     │
             HTTP Requests
                     │
                 FastAPI
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
 File Uploads   Interview APIs   Search APIs
      │
 python-multipart
      │
 aiofiles
      │
      ▼
 ┌──────────────────────────────┐
 │ PDF        Notebook     Image │
 └──────┬─────────┬─────────┬────┘
        │         │         │
 PyMuPDF4LLM  nbformat  PaddleOCR/OCRmyPDF
        │         │         │
        └─────────┴─────────┘
                  │
           Markdown Content
                  │
         markdown-it-py
                  │
         Semantic Chunking
                  │
      sentence-transformers
                  │
            Vector Embeddings
                  │
          NumPy Vector Math
                  │
         Similarity Retrieval
                  │
      transformers (LLMs/Rerankers)
                  │
            JSON Response
             (orjson)
                  │
             FastAPI Response
                  │
                Browser
```

This stack forms a production-ready RAG architecture where each library
has a focused responsibility, making the system modular, maintainable,
and scalable.
