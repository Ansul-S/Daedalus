"""Evaluation of search and of the grader.

The grader's suite uses DeepEval offline: it sends nothing and reads no .env file. Both are
set here, before any module of the package can import it.
"""

import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
