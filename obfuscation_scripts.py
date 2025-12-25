import re
from typing import Dict, List, Optional, Tuple


_DEF_RE = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*(?:->\s*([^:]+))?:\s*$", re.M,)

_PARAM_NAME_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::|=|$)")


def extract_def_header(prompt: str) -> Tuple[str, str, Optional[str], int, int]:
    """
    Locate the first function definition line: def name(params)
    """
    m = _DEF_RE.search(prompt)
    if not m:
        raise ValueError("No function definition line found in prompt.")
    func_name = m.group(1)
    params_str = m.group(2)
    ret = m.group(3).strip() if m.group(3) else None
    return func_name, params_str, ret, m.start(), m.end()


def _split_params(params_str: str) -> List[str]:
    """
    Split parameters by commas while respecting nesting ([], (), {}).
    Prevents breaking types like List[Tuple[int,int]] or default tuples.
    """
    out, buf = [], []
    depth = 0
    for ch in params_str:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)

        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def parse_param_name(token: str) -> Optional[str]:
    #Extract the parameter name from a param token, ignoring '*' and '/'.
    token = token.strip()
    if token in {"*", "/"}:
        return None
    m = _PARAM_NAME_RE.match(token)
    return m.group(1) if m else None


def build_param_rename_map(prompt: str, style: str = "arg") -> Dict[str, str]:
    # Build a deterministic mapping old_param -> new_param.
    _, params_str, _, _, _ = extract_def_header(prompt)
    tokens = _split_params(params_str)
    names = [parse_param_name(t) for t in tokens]
    names = [n for n in names if n is not None]

    mapping: Dict[str, str] = {}
    for i, old in enumerate(names):
        new = f"arg{i}" if style == "arg" else f"x{i}"
        mapping[old] = new
    return mapping


def rename_parameters_in_def(prompt: str, mapping: Dict[str, str]) -> str:
    """
    Rename parameter names *only in the function signature*.
    Keeps annotations and defaults intact.
    """
    func_name, params_str, ret, start, end = extract_def_header(prompt)
    tokens = _split_params(params_str)

    new_tokens = []
    for tok in tokens:
        name = parse_param_name(tok)
        if name and name in mapping:
            tok = re.sub(rf"^\s*{re.escape(name)}\b", mapping[name], tok)
        new_tokens.append(tok)

    new_params_str = ", ".join(new_tokens)
    new_header = f"def {func_name}({new_params_str})"
    if ret is not None:
        new_header += f" -> {ret}"
    new_header += ":"

    return prompt[:start] + new_header + prompt[end:]


# Docstring extraction & rewriting


_DOC_RE = re.compile(r'(\n[ \t]*)("""|\'\'\')([\s\S]*?)(\2)')

def extract_first_docstring(prompt: str) -> Optional[Tuple[str, str, str, int, int]]:
    m = _DOC_RE.search(prompt)
    if not m:
        return None
    indent_prefix = m.group(1)         # e.g., "\n    "
    quote = m.group(2)                 # """ or '''
    inner = m.group(3)                 # content inside
    start = m.start(2)
    end = m.end(4)
    return indent_prefix, quote, inner, start, end


def rename_identifiers_in_docstring(prompt: str, mapping: Dict[str, str]) -> str:
    """
    Low-obf: replace parameter mentions inside docstring (word-boundary).
    Keeps doctests intact.
    """
    doc = extract_first_docstring(prompt)
    if not doc:
        return prompt
    indent_prefix, quote, inner, start, end = doc

    new_inner = inner
    for old, new in mapping.items():
        new_inner = re.sub(rf"\b{re.escape(old)}\b", new, new_inner)

    new_block = f"{quote}{new_inner}{quote}"
    return prompt[:start] + new_block + prompt[end:]


# High-obf docstring sanitizer: keep task contract, drop fingerprints

_DOCTEST_LINE_RE = re.compile(r"^\s*>>>\s*")
_EXAMPLE_LIKE_RE = re.compile(
    r"^\s*(e\.g\.|eg\.|example:|examples:|for example|E\.g\.|Eg\.)", re.IGNORECASE
)

def strip_doctest_blocks(text: str) -> str:
    """
    Remove doctest input lines ('>>> ...') and the immediate following output lines.
    Heuristic: after a >>> line, drop subsequent non-empty lines until blank or next >>>.
    """
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _DOCTEST_LINE_RE.match(line):
            i += 1
            # skip expected outputs
            while i < len(lines):
                nxt = lines[i]
                if _DOCTEST_LINE_RE.match(nxt):
                    break
                if nxt.strip() == "":
                    i += 1
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def sanitize_docstring_keep_contract(inner: str, mapping: Dict[str, str]) -> str:
    """
    High-obf: keep enough semantics to solve, remove solution fingerprints:
      - remove doctests
      - remove explicit example lines (E.g., Example:)
      - remove formula-heavy lines (simple heuristic)
      - rename parameter mentions to match renamed signature
      - soften some highly-specific phrasing (lightweight synonym rules)
      - compress to a short contract-style docstring (1-4 lines)
    """
    # drop doctest blocks
    text = strip_doctest_blocks(inner)

    # drop obvious example lines
    lines = text.splitlines()
    kept: List[str] = []
    for ln in lines:
        if _EXAMPLE_LIKE_RE.match(ln.strip()):
            continue
        # Drop lines that look like "MAD = average | x - x_mean |" etc.
        # Heuristic: has '=' and at least one math-ish symbol.
        if "=" in ln and any(sym in ln for sym in ["|", "*", "+", "-", "/", "^"]):
            continue
        kept.append(ln)

    text = "\n".join(kept)

    # rename parameter mentions
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)

    # light synonym softening (keeps solvable, reduces "canonical fingerprint")
    # Keep these conservative; we do NOT want to destroy task meaning.
    synonym_rules = [
        (r"\bbank account\b", "running total"),
        (r"\bbalance\b", "cumulative value"),
        (r"\bMean Absolute Deviation\b", "a dispersion measure"),
        (r"\bMAD\b", "dispersion value"),
        (r"\bnested parentheses\b", "parenthesis structure"),
        (r"\bdeepest level of nesting\b", "maximum nesting depth"),
        (r"\bdecomposed\b", "split"),
        (r"\binteger part\b", "whole-number part"),
        (r"\bdecimal part\b", "fractional part"),
    ]
    for pat, repl in synonym_rules:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    # normalize whitespace
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # contract-style compression:
    # Keep the first 1–3 non-empty sentences/lines that describe behavior.
    # This preserves task solvability but removes verbose guidance.
    # Strategy:
    #   - prefer non-empty lines
    #   - stop after we have ~3 lines or ~300 chars
    out_lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        out_lines.append(s)
        if len(out_lines) >= 3:
            break
        if sum(len(x) for x in out_lines) >= 300:
            break

    if not out_lines:
        out_lines = ["Implement the function described by its signature."]

    # Ensure ends with period-ish
    compact = " ".join(out_lines)
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact and compact[-1] not in ".!?":
        compact += "."

    return compact


def replace_docstring_with_sanitized(prompt: str, sanitized_text: str) -> str:
    """
    Replace the first docstring content with sanitized_text.
    If docstring doesn't exist, insert it right after def line.
    Preserves indentation.
    """
    def_match = _DEF_RE.search(prompt)
    if not def_match:
        raise ValueError("No function definition line found in prompt.")

    # Infer indentation for function body
    after_def = prompt[def_match.end():]
    indent_match = re.search(r"\n([ \t]+)", after_def)
    indent = indent_match.group(1) if indent_match else "    "

    # Build sanitized docstring block
    new_doc = (
        indent + '"""\n'
        + indent + sanitized_text.strip() + "\n"
        + indent + '"""'
    )

    doc = extract_first_docstring(prompt)
    if doc:
        indent_prefix, quote, _, start, end = doc
        # Keep existing indentation before opening quotes
        existing_indent = indent_prefix.split("\n")[-1]

        replacement = (
            existing_indent + '"""\n'
            + existing_indent + sanitized_text.strip() + "\n"
            + existing_indent + '"""'
        )

        return prompt[:start] + replacement + prompt[end:]
    else:
        # Insert docstring right after def line
        insert_at = def_match.end()
        insertion = "\n" + new_doc + "\n"
        return prompt[:insert_at] + insertion + prompt[insert_at:]


# Formatting helpers + final low/high obfuscators

def normalize_whitespace(prompt: str) -> str:
    """Remove trailing spaces and collapse 3+ blank lines to 2."""
    prompt = "\n".join([ln.rstrip() for ln in prompt.splitlines()])
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip() + "\n"


def low_obfuscate(prompt: str) -> str:
    """
    Low obfuscation:
      - rename parameters in signature
      - optionally update docstring parameter mentions
      - keep doctests/examples
    """
    mapping = build_param_rename_map(prompt, style="arg")
    out = rename_parameters_in_def(prompt, mapping)
    out = rename_identifiers_in_docstring(out, mapping)
    return normalize_whitespace(out)


def high_obfuscate(prompt: str) -> str:
    """
    High obfuscation:
      - rename parameters in signature
      - sanitize docstring:
          * remove doctests/examples
          * keep short contract text (solvable)
          * soften named/fingerprint phrases lightly
    """
    mapping = build_param_rename_map(prompt, style="arg")
    out = rename_parameters_in_def(prompt, mapping)

    doc = extract_first_docstring(out)
    if doc:
        _, _, inner, _, _ = doc
        sanitized = sanitize_docstring_keep_contract(inner, mapping)
        out = replace_docstring_with_sanitized(out, sanitized)
    else:
        out = replace_docstring_with_sanitized(out, "Implement the function described by its signature.")

    return normalize_whitespace(out)