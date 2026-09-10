#!/usr/bin/env python3
"""Audit printed TeX claims against read-only sealed JSON artifacts (Python 3.9+).

Both input locations are mandatory. The adjacent JSON manifest is the default;
--tier1-only does not need a manifest. No input files are ever written.

Manifest schema 1: sources name a run/model pair; tables contain rows of explicit
cells (id, json_path, literal). A row's numeric/sign/verdict tokens after its
first ampersand bind cells by position, so swaps and stale duplicate values fail.
Claims and derived entries use selectors: text between start/end, optionally
within scope, and zero-based token slot; alternatively a regex with a named
"value" group. Selectors must resolve uniquely. Whitespace is normalized only
for locating anchors. TeX formatting is masked with offsets preserved.
Literal precision determines half-unit rounding tolerance unless overridden.
Unresolved claims are deliberate failures, never exemptions. Formula operations
are a closed declarative vocabulary, with no eval or executable expressions.

Tier 1 deliberately checks ONLY direct artifact membership, including derived
four-place quotes; those may be reported again (with attribution) in Tier 3.
Passing Tier 1 alone is not evidence of correct run/model attribution.
"""

import argparse
import bisect
from decimal import Decimal
import json
from pathlib import Path
import re
import sys


NUMBER = r"[+-]?\d+(?:\.\d+)?"
TOKENS = re.compile(r"(?<![\w.])(?:" + NUMBER + r"|[+-]|PASS|FAIL)(?!\w|\.\d)")
FOUR = re.compile(r"(?<![\w.+-])[+-]?\d+\.\d{4}(?!\w|\.\d)")
FIELDS = {"auroc", "ci", "delta", "delta_ci", "auroc_a", "auroc_b"}


def blank(text):
    return "".join("\n" if c == "\n" else " " for c in text)


def clean_tex(text):
    """Mask ignored constructs while preserving every character and line offset."""
    out = list(text)
    i = 0
    while i < len(text):
        end = None
        if text[i] == "%":
            end = text.find("\n", i)
            if end == -1:
                end = len(text)
        elif text[i] == "\\":
            env = re.match(r"\\begin\s*\{(verbatim\*?)\}", text[i:])
            cmd = re.match(r"\\(?:label|ref|Cref|cref|Ref)\*?\s*\{", text[i:])
            if env:
                closing = re.search(r"\\end\s*\{" + re.escape(env[1]) + r"\}",
                                    text[i + env.end():])
                end = (i + env.end() + closing.end()) if closing else len(text)
            elif cmd:
                end = i + cmd.end()
                depth = 1
                while end < len(text) and depth:
                    if text[end] == "\\":
                        end += 2
                        continue
                    depth += (text[end] == "{") - (text[end] == "}")
                    end += 1
            else:
                # Consume control symbols, notably escaped % and paired \\.
                control = re.match(r"\\(?:[A-Za-z]+|[^\n])", text[i:])
                i += control.end() if control else 1
                continue
        if end is not None:
            out[i:end] = blank(text[i:end])
            i = end
        else:
            i += 1
    return "".join(out)


def formatted_mask(text):
    text = re.sub(r"\\[A-Za-z]+\*?", lambda m: " " * len(m[0]), text)
    return re.sub(r"[{}$]", " ", text)


def normalized(text):
    chars, offsets = [], []
    for i, c in enumerate(text):
        if c.isspace():
            if chars and chars[-1] == " ":
                continue
            c = " "
        chars.append(c)
        offsets.append(i)
    offsets.append(len(text))
    return "".join(chars), offsets


def unique_span(text, needle, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    hits = list(re.finditer(re.escape(needle), text[lo:hi]))
    if len(hits) != 1:
        raise ValueError("anchor %r: expected 1 occurrence, found %d" % (needle, len(hits)))
    return lo + hits[0].start(), lo + hits[0].end()


class Document:
    def __init__(self, text):
        self.raw = text
        self.clean = clean_tex(text)
        self.norm, self.offsets = normalized(self.clean)
        self.lines = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(self, offset):
        return bisect.bisect_right(self.lines, offset)

    def context(self, offset):
        line = self.line(offset)
        return " ".join(self.raw.splitlines()[max(0, line - 2):line + 1]).strip()

    def locate(self, selector):
        lo, hi = 0, len(self.norm)
        if "scope" in selector:
            _, lo = unique_span(self.norm, " ".join(selector["scope"].split()))
            if "scope_end" in selector:
                hi = self.norm.find(selector["scope_end"], lo)
                if hi < 0:
                    raise ValueError("missing scope_end")
        if "pattern" in selector:
            pattern = selector["pattern"].replace("NUMBER", NUMBER)
            matches = list(re.finditer(pattern, self.norm[lo:hi]))
            if len(matches) != 1:
                raise ValueError("value pattern: expected 1 match, found %d" % len(matches))
            m = matches[0]
            a, b = lo + m.start("value"), lo + m.end("value")
            return m["value"], self.offsets[a], self.offsets[b]
        _, start = unique_span(self.norm, " ".join(selector["start"].split()), lo, hi)
        end = self.norm.find(" ".join(selector["end"].split()), start, hi)
        if end < 0:
            raise ValueError("missing end anchor %r" % selector["end"])
        a, b = self.offsets[start], self.offsets[end]
        matches = list(TOKENS.finditer(formatted_mask(self.clean[a:b])))
        if "token_count" in selector and len(matches) != selector["token_count"]:
            raise ValueError("expected %d cell tokens, found %d: %s" %
                             (selector["token_count"], len(matches), [m[0] for m in matches]))
        slot = selector.get("slot", 0)
        if not isinstance(slot, int) or slot < 0 or slot >= len(matches):
            raise ValueError("missing token slot %r" % slot)
        m = matches[slot]
        return m[0], a + m.start(), a + m.end()

    def hint(self, selector):
        for key in ("start", "scope"):
            needle = " ".join(selector.get(key, "").split())
            pos = self.norm.find(needle) if needle else -1
            if pos >= 0:
                return self.line(self.offsets[pos])
        return "unlocated (claim removed or anchor changed)"


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON: " + value)

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_float=Decimal,
                         parse_constant=reject_constant, object_pairs_hook=unique_keys)


def number(value):
    if isinstance(value, bool):
        raise ValueError("boolean used as number")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite number")
    return result


def at_path(obj, path):
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


class Bank:
    def __init__(self, root):
        self.runs = {}
        self.values = []
        files = sorted(root.glob("*/*/sealed_gate.json"))
        if not files:
            raise ValueError("no sealed_gate.json files in the supplied runs directory")
        for path in files:
            data = read_json(path)
            key = path.parent.relative_to(root).as_posix()
            self.runs[key] = data
            for model in data["per_model"]:
                for block_name, block in model.items():
                    if not isinstance(block, dict):
                        continue
                    for field in FIELDS & block.keys():
                        vals = block[field] if field in {"ci", "delta_ci"} else [block[field]]
                        if not isinstance(vals, list) or (field in {"ci", "delta_ci"} and len(vals) != 2):
                            raise ValueError("malformed interval in " + str(path))
                        for index, value in enumerate(vals):
                            suffix = ".%d" % index if field in {"ci", "delta_ci"} else ""
                            self.values.append((number(value), "%s/%s/%s.%s%s" %
                                                (key, model["model"], block_name, field, suffix)))
        if not self.values:
            raise ValueError("artifacts contain no comparable values")

    def resolve(self, sources, source, path):
        spec = sources[source]
        data = self.runs[spec["run"]]
        if path.startswith("$."):
            return at_path(data, path[2:])
        models = [m for m in data["per_model"] if spec["model_match"] in m["model"]]
        if len(models) != 1:
            raise ValueError("source %s: expected one model, found %d" % (source, len(models)))
        return at_path(models[0], path)


def formula(expr, bank, sources):
    if isinstance(expr, (int, Decimal)):
        return number(expr)
    if isinstance(expr, str):
        source, path = expr.split(":", 1)
        return number(bank.resolve(sources, source, path))
    op = expr["op"]
    args = [formula(a, bank, sources) for a in expr["args"]]
    if op == "sub" and len(args) == 2:
        return args[0] - args[1]
    if op == "div" and len(args) == 2:
        return args[0] / args[1]
    if op == "mul" and len(args) == 2:
        return args[0] * args[1]
    if op == "sqrt" and len(args) == 1:
        return args[0].sqrt()
    if op == "unoriented" and len(args) == 2:
        if args[1] != -1:
            raise ValueError("unoriented formula requires sign=-1; artifact sign=%s" % args[1])
        return 1 - args[0]
    raise ValueError("unsupported formula or arity: " + op)


def agrees(actual, literal, tolerance=None):
    if literal in {"PASS", "FAIL"}:
        return actual == literal
    expected = number({"+": "1", "-": "-1"}.get(literal, literal))
    if tolerance is None:
        tolerance = Decimal(5).scaleb(-len(literal.split(".")[1]) - 1) if "." in literal else Decimal(0)
    tolerance = number(tolerance)
    if tolerance < 0:
        raise ValueError("negative tolerance")
    # Sealed JSON contains binary-float serialization tails at rounding ties.
    # This slack is far below any published precision; exact signs/counts stay exact.
    slack = Decimal("1e-12") if tolerance else Decimal(0)
    return abs(number(actual) - expected) <= tolerance + slack


def expand(manifest):
    claims = list(manifest["claims"])
    for table in manifest["tables"]:
        for row in table["rows"]:
            for slot, cell in enumerate(row["cells"]):
                entry = dict(cell)
                entry["id"] = table["id"] + "." + row["id"] + "." + cell["id"]
                entry["source"] = row["source"]
                entry["selector"] = {"scope": table["scope"], "scope_end": "\\end{table}",
                                     "start": row["tex_context"] + " &", "end": "\\\\",
                                     "slot": slot, "token_count": len(row["cells"])}
                claims.append(entry)
    ids = [entry["id"] for entry in claims + manifest["derived"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate manifest claim IDs")
    if not claims or not manifest["derived"]:
        raise ValueError("empty claim or derived manifest")
    return claims


def check_claim(doc, entry, bank, sources, derived=False):
    errors = []
    literal = entry["literal"]
    selector = entry["selector"]
    line = doc.hint(selector)
    observed = "<missing>"
    try:
        observed, a, _ = doc.locate(selector)
        line = doc.line(a)
    except (ValueError, IndexError) as exc:
        errors.append("tex=<unlocated>; manifest literal=%r; selector: %s" % (literal, exc))
    if observed != literal:
        errors.append("tex=%r; manifest literal=%r" % (observed, literal))
    try:
        if "unresolved" in entry:
            raise ValueError(entry["unresolved"])
        if derived:
            actual = formula(entry["formula"], bank, sources)
        elif entry.get("transform") == "e18_verdict":
            auroc = number(bank.resolve(sources, entry["source"], "E18_sealed_rank1_lowrank32.auroc"))
            lower = number(bank.resolve(sources, entry["source"], "E18_sealed_rank1_lowrank32.ci.0"))
            threshold = number(bank.resolve(sources, entry["source"], "$.sealed_spec.threshold"))
            actual = "PASS" if auroc >= threshold and lower > Decimal("0.5") else "FAIL"
        else:
            actual = bank.resolve(sources, entry["source"], entry["json_path"])
        if not agrees(actual, literal, entry.get("tolerance")):
            errors.append("published=%s; %s=%s; tolerance=%s" %
                          (literal, "computed" if derived else "artifact", actual,
                           entry.get("tolerance", "half unit at printed precision")))
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError) as exc:
        errors.append("published=%s; artifact/computed=<unresolved>: %s" % (literal, exc))
    return ["line %s [%s]: %s" % (line, entry["id"], error) for error in errors]


def tier1(doc, bank):
    errors = []
    # Every quantity in the artifact bank is an AUROC, a CI endpoint, or a
    # difference of AUROCs, so all of them live in [-1, 1]. Restricting the scan
    # to that range costs no coverage and drops the one systematic false
    # positive: DOI suffixes such as 10.1038 are 4-dp decimals too.
    matches = [m for m in FOUR.finditer(doc.clean) if abs(number(m[0])) <= 1]
    for m in matches:
        value = number(m[0])
        nearest, source = min(bank.values, key=lambda pair: abs(pair[0] - value))
        if abs(nearest - value) > Decimal("0.00005"):
            errors.append("line %d: tex=%s; nearest artifact=%s (%s); context: %s" %
                          (doc.line(m.start()), m[0], nearest, source, doc.context(m.start())))
    return len(matches), errors


def self_test(doc, bank, claims, derived, sources):
    # Sentinel must be exactly 4 dp, inside the [-1, 1] window Tier 1 scans, and
    # further than the match tolerance from every banked value. The bank is
    # finite, so a scan of the 4-dp grid is guaranteed to find one.
    sentinel = None
    for step in range(1, 10000):
        candidate = Decimal(step).scaleb(-4)
        if all(abs(candidate - v) > Decimal("0.00005") for v, _ in bank.values):
            sentinel = format(candidate, ".4f")
            break
    if sentinel is None:
        raise ValueError("self-test: no 4-dp value in [0, 1] is absent from the bank")
    injected = Document(doc.raw + "\nChecker injected value: " + sentinel + "\n")
    _, failures = tier1(injected, bank)
    target_line = injected.line(len(doc.raw) + 1)
    if not any("line %d:" % target_line in f and "tex=" + sentinel in f for f in failures):
        raise ValueError("self-test: Tier 1 failed to detect injected value " + sentinel)
    # Also reproduce the historical failure: a valid number from the wrong run.
    targets = [c for c in claims if c["id"] == "baselines.llama.v3"]
    if targets:
        entry = targets[0]
        _, start, end = doc.locate(entry["selector"])
        changed = Document(doc.raw[:start] + "0.8975" + doc.raw[end:])
        if not any("tex='0.8975'" in f for f in check_claim(changed, entry, bank, sources)):
            raise ValueError("self-test: Tier 2 missed wrong-run Llama regression")
    derived_targets = [c for c in derived if c["id"] == "bar.multiple"]
    if derived_targets:
        entry = derived_targets[0]
        _, start, end = doc.locate(entry["selector"])
        changed = Document(doc.raw[:start] + "8.9" + doc.raw[end:])
        if not any("tex='8.9'" in f for f in check_claim(changed, entry, bank, sources, True)):
            raise ValueError("self-test: Tier 3 missed changed published multiple")
    fixture = Document("% 9.8765\n\\label{9.8765} \\ref{9.8765} \\Cref{9.8765}\n"
                       "\\begin{verbatim}\n9.8765\n\\end{verbatim}\n"
                       "\\% +9.8765. 0.12345\n")
    if [m[0] for m in FOUR.finditer(fixture.clean)] != ["+9.8765"]:
        raise ValueError("self-test: TeX masking / exact precision failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True, help="TeX input file")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Directory containing dated run directories")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("paper_numbers_manifest.json"))
    parser.add_argument("--tier1-only", action="store_true", help="Scan direct four-place quotes without a manifest")
    parser.add_argument("--self-test", action="store_true", help="Check injected failures in memory, then audit original input")
    args = parser.parse_args()
    try:
        doc = Document(args.tex.read_text(encoding="utf-8"))
        bank = Bank(args.runs_dir)
        count1, errors1 = tier1(doc, bank)
        claims, derived, sources = [], [], {}
        if not args.tier1_only:
            manifest = read_json(args.manifest)
            if manifest["schema_version"] != 1:
                raise ValueError("unsupported manifest schema")
            claims = expand(manifest)
            derived, sources = manifest["derived"], manifest["sources"]
        errors2 = [e for c in claims for e in check_claim(doc, c, bank, sources)]
        errors3 = [e for c in derived for e in check_claim(doc, c, bank, sources, True)]
        if args.self_test:
            self_test(doc, bank, claims, derived, sources)
        for name, errors in (("Tier 1 — unattributed precision", errors1),
                             ("Tier 2 — explicit claims", errors2),
                             ("Tier 3 — derived values", errors3)):
            if errors:
                print(name)
                for error in errors:
                    print("  " + error)
        failed = bool(errors1 or errors2 or errors3)
        print("%s: tier1=%d quotes, tier2=%d claims, tier3=%d derived; %d mismatches%s" %
              ("FAIL" if failed else "OK", count1, len(claims), len(derived),
               len(errors1) + len(errors2) + len(errors3),
               "; self-test passed" if args.self_test else ""))
        return int(failed)
    except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError, re.error) as exc:
        print("FAIL: input/manifest/self-test error; tex line=n/a, expected=valid input, actual=%s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
