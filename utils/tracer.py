"""
Agent I/O Tracer
=================
Writes structured runtime input/output logs for each agent call
to a single text file for post-hoc inspection.

Usage:
    from utils.tracer import Tracer, set_global_tracer, get_tracer
    tracer = Tracer("agent_io_log.txt")
    set_global_tracer(tracer)
    tracer.log_section("ROUND 1 — AGENT 1: PROPOSAL")
    tracer.log_kv("Strategy", "exploration")
    tracer.log_archs("Proposals", proposals)
    tracer.close()
"""

import json
import os
from datetime import datetime

W = 82  # total line width


class Tracer:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self._write_header()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_int(n):
        """Format integer with comma separators: 1234567 → '1,234,567'."""
        if n is None:
            return "None"
        return f"{int(n):,}"

    @staticmethod
    def _fmt_arch(arch):
        """Compact one-liner for an architecture dict."""
        return (f"embed={arch.get('embed_dim','?'):<4} "
                f"depth={arch.get('depth','?'):<2} "
                f"mlp={arch.get('mlp_ratio','?'):<2} "
                f"heads={arch.get('num_heads','?')}")

    @staticmethod
    def _fmt_float(v, width=7):
        """Format a float to 4 decimal places, right-aligned."""
        if v is None:
            return "  None".ljust(width)
        return f"{float(v):.4f}".rjust(width)

    def _w(self, text=""):
        """Write a line (no trailing newline — caller decides)."""
        self.f.write(text)

    def _wl(self, text=""):
        """Write a line + newline."""
        self.f.write(text + "\n")

    # ------------------------------------------------------------------
    #  Header / Footer  (double-line box)
    # ------------------------------------------------------------------
    def _write_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._wl("╔" + "═" * (W - 2) + "╗")
        self._wl("║  " + "MAS-NAS Agent I/O Runtime Log".ljust(W - 4) + "║")
        self._wl("║  " + f"Started: {ts}".ljust(W - 4) + "║")
        self._wl("╚" + "═" * (W - 2) + "╝")
        self._wl()
        self.f.flush()

    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._wl()
        self._wl("╔" + "═" * (W - 2) + "╗")
        self._wl("║  " + f"Finished: {ts}".ljust(W - 4) + "║")
        self._wl("╚" + "═" * (W - 2) + "╝")
        self.f.close()

    # ------------------------------------------------------------------
    #  Section header  (single-line box with timestamp)
    # ------------------------------------------------------------------
    def log_section(self, title):
        ts = datetime.now().strftime("%H:%M:%S")
        inner = W - 2
        # title left, timestamp right
        padding = inner - len(title) - len(ts) - 4  # 2 spaces each side
        if padding < 1:
            padding = 1
        content = f"  {title}" + " " * padding + f"{ts}  "
        # ensure exact width
        content = content[:inner].ljust(inner)
        self._wl()
        self._wl("┌" + "─" * inner + "┐")
        self._wl("│" + content + "│")
        self._wl("└" + "─" * inner + "┘")
        self._wl()
        self.f.flush()

    # ------------------------------------------------------------------
    #  Subsection  (lightweight separator)
    # ------------------------------------------------------------------
    def log_subsection(self, title):
        self._wl(f"  ── {title} ──")
        self.f.flush()

    # ------------------------------------------------------------------
    #  Key-Value pair  (aligned)
    # ------------------------------------------------------------------
    def log_kv(self, key, value):
        """Write an aligned key : value line."""
        k = str(key).ljust(18)
        if isinstance(value, int) and not isinstance(value, bool):
            v = self._fmt_int(value)
        else:
            v = str(value)
        self._wl(f"  {k}: {v}")
        self.f.flush()

    # ------------------------------------------------------------------
    #  Architecture formatting
    # ------------------------------------------------------------------
    def log_arch(self, idx, arch):
        """Log a single architecture with index (1-based)."""
        line = f"  [{idx}] {self._fmt_arch(arch)}"
        rationale = arch.get("rationale")
        if rationale:
            self._wl(line)
            # wrap rationale to ~70 chars
            self._wl(f"      \"{rationale[:200]}\"")
        else:
            self._wl(line)
        self.f.flush()

    def log_archs(self, label, archs):
        """Log a list of architectures with a label header."""
        if not archs:
            self._wl(f"  {label}: (none)")
            self.f.flush()
            return
        self._wl(f"  {label} ({len(archs)}):")
        for i, a in enumerate(archs, 1):
            self.log_arch(i, a)
        self.f.flush()

    def log_arch_table(self, archs, label="Results"):
        """Log architectures with metrics in aligned columns."""
        if not archs:
            self._wl(f"  {label}: (none)")
            self.f.flush()
            return
        # Determine which metric columns exist
        metric_cols = []
        candidate_cols = [
            "num_params", "flops",
            "val_accuracy", "val_f1", "val_auroc", "val_auprc",
            "accuracy", "f1", "auroc", "auprc",
        ]
        for col in candidate_cols:
            if any(col in a for a in archs):
                metric_cols.append(col)

        # Header
        hdr = "    #   embed  depth  mlp  heads"
        col_labels = {
            "num_params": "params",
            "flops": "FLOPs",
            "val_accuracy": "Acc",
            "val_f1": "F1",
            "val_auroc": "AUROC",
            "val_auprc": "AUPRC",
            "accuracy": "Acc",
            "f1": "F1",
            "auroc": "AUROC",
            "auprc": "AUPRC",
        }
        for col in metric_cols:
            lbl = col_labels.get(col, col)
            if col in ("num_params", "flops"):
                hdr += f"  {lbl:>11}"
            else:
                hdr += f"  {lbl:>7}"
        self._wl(f"  {label}:")
        self._wl(hdr)

        # Rows
        for i, a in enumerate(archs, 1):
            row = (f"    {i:<4}"
                   f"{a.get('embed_dim','?'):>5}  "
                   f"{a.get('depth','?'):>5}  "
                   f"{a.get('mlp_ratio','?'):>3}  "
                   f"{a.get('num_heads','?'):>5}")
            for col in metric_cols:
                v = a.get(col)
                if col in ("num_params", "flops"):
                    row += f"  {self._fmt_int(v):>11}"
                else:
                    row += f"  {self._fmt_float(v)}"
            self._wl(row)
        self.f.flush()

    # ------------------------------------------------------------------
    #  Rejected proposals (with critique + tags)
    # ------------------------------------------------------------------
    def log_rejected(self, rejected_list, label="Rejected"):
        """Log rejected proposals with critique and risk tags."""
        if not rejected_list:
            self._wl(f"  {label}: (none)")
            self.f.flush()
            return
        self._wl(f"  {label} ({len(rejected_list)}):")
        for i, r in enumerate(rejected_list, 1):
            prop = r.get("proposal", r)
            self._wl(f"    [{i}] {self._fmt_arch(prop)}")
            critique = r.get("critique", "")
            if critique:
                self._wl(f"        Critique: \"{critique[:300]}\"")
            tags = r.get("risk_tags", [])
            if tags:
                self._wl(f"        Tags: {tags}")
        self.f.flush()

    # ------------------------------------------------------------------
    #  SHAP importance
    # ------------------------------------------------------------------
    def log_shap(self, shap_dict, top_n=10):
        """Log SHAP feature importance, sorted descending."""
        if not shap_dict:
            self._wl("  SHAP importance: (not available)")
            self.f.flush()
            return
        sorted_items = sorted(shap_dict.items(), key=lambda x: -abs(float(x[1])))
        n = min(top_n, len(sorted_items))
        self._wl(f"  SHAP importance (top {n}):")
        for name, val in sorted_items[:n]:
            self._wl(f"    {str(name).ljust(18)}: {float(val):.4f}")
        if len(sorted_items) > n:
            self._wl(f"    ... ({len(sorted_items) - n} more features)")
        self.f.flush()

    # ------------------------------------------------------------------
    #  LLM Prompt / Response  (from inside agents)
    # ------------------------------------------------------------------
    def log_prompt(self, prompt_text):
        """Log the full LLM prompt (truncated for readability)."""
        max_len = 3000
        total = len(prompt_text)
        if total <= max_len:
            display = prompt_text
        else:
            display = prompt_text[:max_len] + f"\n    ... (truncated, total {total:,} chars)"
        self._wl(f"  [PROMPT] ({total:,} chars):")
        for line in display.split("\n"):
            self._wl(f"    {line}")
        self.f.flush()

    def log_response(self, response_text):
        """Log the raw LLM response."""
        total = len(response_text)
        self._wl(f"  [RESPONSE] ({total:,} chars):")
        for line in response_text.split("\n"):
            self._wl(f"    {line}")
        self.f.flush()

    # ------------------------------------------------------------------
    #  Free-text note
    # ------------------------------------------------------------------
    def log_note(self, text):
        """Log a free-text note."""
        self._wl(f"  [NOTE] {text}")
        self.f.flush()

    # ------------------------------------------------------------------
    #  Fallback: generic input/output (for any data not covered above)
    # ------------------------------------------------------------------
    def log_input(self, label, data):
        """Fallback: log an input field as JSON."""
        self._wl(f"  [INPUT] {label}:")
        self._write_data(data, indent=4)
        self.f.flush()

    def log_output(self, label, data):
        """Fallback: log an output field as JSON."""
        self._wl(f"  [OUTPUT] {label}:")
        self._write_data(data, indent=4)
        self.f.flush()

    def _write_data(self, data, indent=4):
        prefix = " " * indent
        if isinstance(data, (dict, list)):
            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            for line in text.split("\n"):
                self._wl(f"{prefix}{line}")
        else:
            self._wl(f"{prefix}{data}")


# ------------------------------------------------------------------
#  Global singleton — set by mas_search.py, read by agents
# ------------------------------------------------------------------
_global_tracer = None


def set_global_tracer(tracer):
    global _global_tracer
    _global_tracer = tracer


def get_tracer():
    return _global_tracer
