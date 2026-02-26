# Investigation: CI Lint, Format, Type-Check, and Security Failures

## Bug Summary

Three CI checks are failing:
1. **`ruff format --check`** — 2 files have formatting violations
2. **`mypy --strict`** — missing type stubs for `torch` module
3. **`bandit -r src/ -q`** — 2 uses of `assert` in production code

## Root Cause Analysis

### Issue 1: Ruff Formatting (2 files)

**Files affected:**
- `src/qr_sampler/entropy/quantum.py`
- `tests/test_entropy/test_quantum.py`

**Root cause:** Line-wrapping style doesn't match ruff's formatter expectations. The ruff formatter prefers:
- Collapsing short expressions onto one line when they fit within 100 chars
- Expanding function call arguments to one-per-line when a trailing comma is present but the expression exceeds line length

**Specific changes needed in `quantum.py`:**
1. Line 146-148: Collapse the `raise EntropyUnavailableError(...)` into a single line (fits in 100 chars)
2. Line 393-394: Expand the `self._unary_method(...)` call arguments to one-per-line

**Specific changes needed in `test_quantum.py`:**
- 6 instances of `AsyncMock(return_value=_encode_mock_response(...))` and `call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")` that need collapsing to single lines

**Fix:** Run `ruff format src/qr_sampler/entropy/quantum.py tests/test_entropy/test_quantum.py`

### Issue 2: Mypy `torch` Import (1 file)

**File affected:** `src/qr_sampler/processor.py:262`

**Root cause:** `processor.py` conditionally imports `torch` at line 262 (inside `_create_onehot_template`). In CI, `torch` is not installed because it's an implicit dependency provided by the vLLM runtime. The `pyproject.toml` already has a `[[tool.mypy.overrides]]` section to ignore missing imports for `grpc` and `grpc.*`, but no such override exists for `torch`.

**Fix:** Add a mypy override in `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = ["torch", "torch.*"]
ignore_missing_imports = true
```

### Issue 3: Bandit `assert_used` (1 file, 2 locations)

**File affected:** `src/qr_sampler/entropy/quantum.py`

**Locations:**
- Line 409: `assert self._stream_method is not None  # validated in __init__`
- Line 427: `assert self._stream_method is not None  # validated in __init__`

**Root cause:** `assert` statements are removed when Python runs with `-O` (optimized bytecode). Bandit flags this as B101 because production code should not rely on `assert` for runtime invariant checks. Both asserts guard `self._stream_method` before use in streaming methods. The `__init__` constructor validates that `_stream_method` is set for streaming modes, so these asserts are defensive checks that should still be present under `-O`.

**Fix:** Replace the `assert` statements with explicit `if ... is None: raise` checks. Since the condition being violated would indicate a programming error (not user input), `RuntimeError` is appropriate. However, `EntropyUnavailableError` also works since callers already handle that.

## Affected Components

| Component | File | Issue |
|-----------|------|-------|
| Entropy / quantum gRPC source | `src/qr_sampler/entropy/quantum.py` | Formatting + bandit |
| Entropy / quantum gRPC tests | `tests/test_entropy/test_quantum.py` | Formatting |
| Processor | `src/qr_sampler/processor.py` | mypy (indirectly, via pyproject.toml) |
| Build config | `pyproject.toml` | Missing torch mypy override |

## Proposed Solution

### Step 1: Fix formatting
Run `ruff format` on the two files. This is a pure whitespace/style change with no behavioral impact.

### Step 2: Add torch mypy override
Add to `pyproject.toml`:
```toml
[[tool.mypy.overrides]]
module = ["torch", "torch.*"]
ignore_missing_imports = true
```
This follows the existing pattern used for the `grpc` module.

### Step 3: Replace assert with explicit checks
In `quantum.py`, replace:
```python
assert self._stream_method is not None  # validated in __init__
```
with:
```python
if self._stream_method is None:  # pragma: no cover — validated in __init__
    raise EntropyUnavailableError("Stream method not initialized")
```

This satisfies bandit (no `assert` in production code) while preserving the defensive check.

## Edge Cases and Side Effects

- **Formatting:** Pure style changes, no behavioral change. Tests will pass identically.
- **Mypy override:** Only affects type checking, no runtime impact. This is the standard approach for optional dependencies.
- **Assert replacement:** The `EntropyUnavailableError` is already caught by `FallbackEntropySource`, so if this impossible condition ever occurred, the fallback chain would handle it gracefully instead of a raw `AssertionError`.
- **No test changes needed** for the bandit fix — the assert was never hit in tests (it guards an invariant from `__init__`).

## Implementation Notes

### Changes Made

1. **`src/qr_sampler/entropy/quantum.py`** — Replaced 2 `assert self._stream_method is not None` statements (lines 409 and 427) with explicit `if ... is None: raise EntropyUnavailableError(...)` checks. Added `# pragma: no cover` since these branches are unreachable by design (validated in `__init__`). Ruff format also applied.

2. **`tests/test_entropy/test_quantum.py`** — Ruff format applied (whitespace/line-wrapping changes only, no behavioral changes).

3. **`pyproject.toml`** — Added `[[tool.mypy.overrides]]` section for `["torch", "torch.*"]` with `ignore_missing_imports = true`, following the existing pattern for `grpc`.

### Test Results

- `ruff format --check src/ tests/` — **PASS** (52 files already formatted)
- `ruff check src/ tests/` — **PASS** (all checks passed)
- `mypy --strict src/` — **PASS** (no issues found in 30 source files)
- `bandit -r src/ -q` — **PASS** (no issues)
- `pytest tests/ -v` — **PASS** (308 passed, 2 warnings in 3.97s)

The 2 warnings are pre-existing RuntimeWarnings about unawaited coroutines in bidi streaming test teardown (mock cleanup), unrelated to the changes made.
