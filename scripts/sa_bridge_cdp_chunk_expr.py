#!/usr/bin/env python3
"""Emit CDP Runtime.evaluate expression for chunked ChatGPT paste."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHUNKS_JSON = Path("/tmp/sa_chunks.json")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sa_bridge_cdp_chunk_expr.py <-1|0|1|2|3|4>", file=sys.stderr)
        return 2
    step = int(sys.argv[1])
    if step == -1:
        print("(function(){ window._saParts = []; return 'init'; })()")
        return 0
    if step == 4:
        print(
            "(function(){\n"
            "  const msg = window._saParts.join('');\n"
            "  const ta = document.querySelector('#prompt-textarea');\n"
            "  ta.focus();\n"
            "  ta.textContent = msg;\n"
            "  ta.dispatchEvent(new InputEvent('input', {bubbles: true}));\n"
            "  return 'len=' + msg.length;\n"
            "})()"
        )
        return 0
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    b64 = chunks[step]
    print(
        "(function(){\n"
        "  window._saParts = window._saParts || [];\n"
        f"  window._saParts.push(new TextDecoder().decode(Uint8Array.from(atob('{b64}'), c => c.charCodeAt(0))));\n"
        "  return String(window._saParts.length);\n"
        "})()"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
