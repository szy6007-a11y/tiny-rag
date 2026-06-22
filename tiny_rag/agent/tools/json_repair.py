from __future__ import annotations


def repair_json(value: str) -> str:
    """Repair common malformed JSON argument strings produced by LLMs."""
    s = (value or "").strip()
    if s == "":
        return "{}"

    if not s.startswith("{"):
        if ":" in s or "=" in s:
            s = "{" + s + "}"
        else:
            return s

    s = _fix_invalid_escapes(s)
    s = _fix_trailing_commas(s)
    s = _balance_brackets(s)
    return s


def _fix_invalid_escapes(s: str) -> str:
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(s):
        char = s[i]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            out.append(char)
            in_string = False
            i += 1
            continue

        if char != "\\":
            out.append(char)
            i += 1
            continue

        if i + 1 >= len(s):
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]
        if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
            out.append(char)
            out.append(nxt)
        else:
            out.append("\\\\")
            out.append(nxt)
        i += 2

    return "".join(out)


def _fix_trailing_commas(s: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(s):
        if escaped:
            escaped = False
            out.append(char)
            continue
        if char == "\\" and in_string:
            escaped = True
            out.append(char)
            continue
        if char == '"':
            in_string = not in_string
            out.append(char)
            continue
        if in_string:
            out.append(char)
            continue
        if char == ",":
            next_non_space = _find_next_non_space(s, index + 1)
            if next_non_space >= 0 and s[next_non_space] in ("}", "]"):
                continue
        out.append(char)
    return "".join(out)


def _find_next_non_space(s: str, start: int) -> int:
    for index in range(start, len(s)):
        if not s[index].isspace():
            return index
    return -1


def _balance_brackets(s: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in s:
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if stack and stack[-1] == char:
                stack.pop()

    if in_string:
        s += '"'

    while stack:
        s += stack.pop()

    return s


RepairJSON = repair_json
