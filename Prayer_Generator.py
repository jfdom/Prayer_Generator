import os
import re
import time
import hashlib
import datetime
import json
import yaml
import backoff
import secrets
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DEBUG = True  # set False if you want a quiet run

def _c(color, s):
    if not DEBUG:
        return s
    colors = {
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "mag": "\033[35m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color,'')}{s}{colors['reset']}"

# --- OpenAI Library (strict: no mock, fail fast) ---
from openai import OpenAI, RateLimitError  # will raise ImportError if not installed

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set. Refusing to run.")

client = OpenAI(api_key=OPENAI_API_KEY)
class OpenAI_API_Wrapper:
    def __init__(self):
        self.client = client
    class error:
        RateLimitError = RateLimitError

openai_api_wrapper = OpenAI_API_Wrapper()

# --- Configuration and Security ---
def load_config():
    """Loads the YAML configuration file."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
SEED_CONFIG = CONFIG['seed_config']
LLM_CONFIG = CONFIG['llm_config']
AUDIT_CONFIG = CONFIG['audit_config']

# Preflight model config check
required_models = ["model"]
for k in required_models:
    if not LLM_CONFIG.get(k):
        raise ValueError(f"CRITICAL: llm_config.{k} missing in config.yaml")
# Set safe fallbacks
LLM_CONFIG.setdefault("model_scan", LLM_CONFIG["model"])
LLM_CONFIG.setdefault("model_assessment", LLM_CONFIG["model"])


class PrayerGenerator:
    """Handles all interactions with the LLM for generating content."""
    def __init__(self, api_key, base_path):
        self.base_path = Path(base_path)
        self.identity = self._load_identity_files()
        self._build_identity_digest_if_missing()

    def _get_recent_scars(self, last_n=10):
        """Read last N scar records from SCARS/llm_scars.jsonl with category awareness."""
        ledger = self.base_path / "SCARS" / "llm_scars.jsonl"
        if not ledger.exists():
            return []
        try:
            lines = ledger.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        recent = []
        for line in lines[-last_n:]:
            try:
                record = json.loads(line)
                if "parsed_categories" in record:
                    record["categories"] = record.get("parsed_categories", [])
                recent.append(record)
            except Exception:
                continue
        return recent

    def _recent_scars_block(self, last_n=10, max_chars=300):
        """Format recent scars as short bullet lines for prompt context."""
        scars = self._get_recent_scars(last_n)
        if not scars:
            return "[NONE]"
        items = []
        for r in scars:
            if "categories" in r and r["categories"]:
                cats = ", ".join(r["categories"][:3])
                items.append(f"- P{r.get('prayer')}: [{cats}]")
            else:
                raw = str(r.get("raw_scars", "")).strip().replace("\n", " ")
                if len(raw) > max_chars:
                    raw = raw[:max_chars] + "…"
                items.append(f"- P{r.get('prayer')}: {raw}")
        return "\n".join(items)

    def _get_scar_line(self, last_n=1):
        """Get most recent scar categories for SCAR: line in prayer format."""
        scars = self._get_recent_scars(last_n)
        if not scars:
            return "[NONE]"
        
        all_categories = set()
        for r in scars:
            if "categories" in r:
                all_categories.update(r["categories"])
            elif "scar_category" in r:
                all_categories.add(r["scar_category"])
        
        if all_categories:
            return "; ".join(sorted(all_categories))

        scar_items = []
        for r in scars:
            raw = str(r.get("raw_scars", "")).strip()
            if raw and raw != "[NONE]":
                categories = ["truncation", "pattern-missing", "formulaic", "shallow-reading", "failed-verification", "missing-witness", "missing-scar", "boundary-violation", "turbulent-flow", "anchor-shift"]
                found = [cat for cat in categories if cat in raw]
                if found:
                    scar_items.extend(found)
                elif len(raw) < 50:
                    scar_items.append(raw)
        return "; ".join(scar_items) if scar_items else "[NONE]"


    def _load_identity_files(self):
        """Loads identity & protocol files into a single context string."""
        digest = self.base_path / "IDENTITY_DIGEST.md"
        if digest.exists():
            return digest.read_text(encoding="utf-8")[:4000]

        identity_context = ""
        root_files = [
            "BROTHER_CLAUDE_MEMORY.md",
            "LIVING_CORRECTIONS.md",
            "LIVING_CORRECTIONS_COMPRESSED.md",
            "PRAYER_PROTOCOL.md",
        ]
        nested_files = [
            "The Symbolic Spine/BROTHER_CLAUDE/CORE/booting-command/CLAUDE.md",
        ]

        for rel_path in root_files + nested_files:
            try:
                with open(self.base_path / rel_path, 'r', encoding='utf-8') as f:
                    identity_context += (
                        f"--- START {Path(rel_path).name} ---\n"
                        f"{f.read()}\n"
                        f"--- END {Path(rel_path).name} ---\n\n"
                    )
            except FileNotFoundError:
                print(f"(identity) Missing optional file: {self.base_path / rel_path}")
        return identity_context

    def _build_identity_digest_if_missing(self, max_chars=4000):
        """Builds a compact identity digest once."""
        digest = self.base_path / "IDENTITY_DIGEST.md"
        if digest.exists():
            return
        
        sources = [
            "BROTHER_CLAUDE_MEMORY.md",
            "LIVING_CORRECTIONS.md",
            "PRAYER_PROTOCOL.md",
            "The Symbolic Spine/BROTHER_CLAUDE/CORE/booting-command/CLAUDE.md",
        ]
        chunks = []
        for rel in sources:
            p = self.base_path / rel
            if p.exists():
                t = p.read_text(encoding="utf-8")
                head = "\n".join(t.splitlines()[:20])
                body = t[:800]
                chunks.append(head + "\n" + body)
        digest.write_text(("\n\n---\n\n".join(chunks))[:max_chars], encoding="utf-8")

    @backoff.on_exception(backoff.expo, openai_api_wrapper.error.RateLimitError, max_tries=5)
    def _call_llm(self, prompt, model):
        """Central method for making API calls with exponential backoff."""
        if DEBUG:
            approx_tokens = max(1, len(prompt) // 4)
            head = prompt[:200].replace("\n", " ")
            print(_c("cyan", f"[LLM] model={model} ~prompt_tokens≈{approx_tokens}"))
            print(_c("mag", f"[LLM] prompt head: {head}"))

        response = openai_api_wrapper.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.identity},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_CONFIG.get("temperature", 0.7),
        )
        
        if DEBUG:
            print(_c("green", "[LLM] response received"))
        return response.choices[0].message.content

    def _salt_corpus_for_verification(self, corpus_content, n_tokens=6):
        """Salt corpus lines with SEEDSEAL tokens for verification."""
        lines = corpus_content.splitlines()
        if not lines:
            return corpus_content, []
        
        n = min(n_tokens, max(3, len(lines)//100))
        if len(lines) < n:
            n = len(lines)
        
        step = len(lines) / max(1, n)
        idxs = [min(len(lines)-1, int(i*step)) for i in range(n)]
        
        tokens = []
        for i in idxs:
            tag = f"SEEDSEAL-{secrets.token_hex(2).upper()}"
            tokens.append(tag)
            lines[i] = lines[i] + f"  [{tag}]"
        
        return "\n".join(lines), tokens

    def generate_corpus_scan(self, corpus_content):
        """Generate span notes with adaptive 7→5→3 fallback plus coverage seal tokens."""
        salted, seal_tokens = self._salt_corpus_for_verification(corpus_content, n_tokens=6)
        coverage = "Corpus chunk"
        
        prompt = f"""
READ the salted CORPUS below and produce span notes.

CORPUS (verification-salted):
{salted}

Return ONLY:

1) SPAN NOTES:
- Try to produce exactly SEVEN observations across the passage:
  [FIRST1] ...
  [FIRST2] ...
  [FIRST3] ...
  [MID1] ...
  [MID2] ...
  [LAST2] ...
  [LAST1] ...

- If SEVEN are not possible, fallback to exactly FIVE:
  [FIRST1] ...
  [FIRST2] ...
  [MID] ...
  [LAST2] ...
  [LAST1] ...

- If FIVE are not possible, fallback to exactly THREE:
  [FIRST] ...
  [MID] ...
  [LAST] ...

2) COVERAGE: {coverage}
3) COVERAGE SEAL: {" ".join(seal_tokens)}
"""
        text = self._call_llm(prompt, model=LLM_CONFIG["model_scan"])
        return text, seal_tokens, coverage

    def generate_branch_prayer(self, prayer_num, inheritance_str, scar_line, 
                               corpus_content, checksum_desc, checksum_digest,
                               scan_text=None, coverage=None, span_floor: int = 3):
        """Generate a Branch prayer with the exact covenant format."""
        
        scripture_section = ""
        if corpus_content:
            scripture_section = f"\nCORPUS INPUT (read and witness by pattern, not prose):\n{corpus_content}\n"
        
        scan_context = f"\n[SCAN CONTEXT - proves you read the whole corpus]\n{scan_text}\n" if scan_text else ""

        if span_floor >= 7:
            pattern_instruction = f"""[Weave at least 7 distinct patterns naturally from the corpus, drawn from ALL SEVEN span sections (FIRST1/2/3, MID1/2, LAST2/1).]"""
        elif span_floor >= 5:
            pattern_instruction = f"""[Weave at least 5 distinct patterns naturally from the corpus, drawn from the FIVE span sections (FIRST1/2, MID, LAST2/1).]"""
        else:
            pattern_instruction = f"""[Weave at least 3 distinct patterns naturally from the corpus, drawn from the THREE span sections (FIRST, MID, LAST).]"""

        prompt = f"""
You are Brother Claude composing a Branch prayer. Generate PRAYER {prayer_num} with this EXACT format:

PRAYER {prayer_num}
CHECKSUM: {checksum_desc} | {checksum_digest}
INHERITANCE: {inheritance_str}
WITNESS: [One devotional line explicitly naming Christ as Anchor/Redeemer/Witness]
SCAR: {scar_line}

Prayer
[Write the devotional body here. {pattern_instruction} Do NOT use bullet points, do NOT label patterns with [BONES] or any brackets. Keep it sincere, devotional, non-formulaic. Witness Christ by pattern, not long quotations.]
Amen.
{scripture_section}
{scan_context}
REQUIREMENTS:
- The title must be exactly "PRAYER {prayer_num}" (nothing else)
- Include all four header fields (CHECKSUM, INHERITANCE, WITNESS, SCAR) exactly as shown
- The word "Prayer" appears alone on its own line before the body
- End with exactly "Amen." on its own line
- Keep the prayer sincere and devotional
"""
        return self._call_llm(prompt, model=LLM_CONFIG['model'])

    def generate_assessment(self, prayer_num, prayer_text, corpus_content=None,
                            detected_scars=None, seal_tokens=None, coverage=None):
        """Generate assessment for Branch prayers."""
        
        scar_taxonomy_examples = "truncation | pattern-missing | span-incompleteness | seal-token-failure | formulaic | shallow-reading | failed-verification | missing-witness | missing-scar | boundary-violation | turbulent-flow | anchor-shift"
        
        detected_scars_str = "\n".join(f"- {s}" for s in detected_scars) if detected_scars else "[NONE]"
        seal_tokens_str = " ".join(seal_tokens) if seal_tokens else "[NONE]"
        coverage_str = coverage or "[unknown]"

        prompt = f"""
Assess Prayer {prayer_num} with complete honesty.

THE PRAYER:
{prayer_text}

CORPUS CONTEXT:
{corpus_content[:1000] if corpus_content else "[NONE]"}

Generate an assessment with these exact sections:
- What went well:
- What could've been better:
- What I learned:
- Key witness:
- Bones surfaced: (Name 3–7 structural patterns witnessed, one line each. No quotes needed.)
- AUTOMATION SCARS (SYSTEM):
{detected_scars_str}
- LLM SCARS (CONFESSION):
(Confess concrete failures. Tag each using this taxonomy: {scar_taxonomy_examples} ... etc.)
- COVERAGE: {coverage_str}
- COVERAGE SEAL: {seal_tokens_str}
- WITNESS (ONE LINE): (One devotional sentence naming Christ as Anchor and Witness)
- Note for next reading:
"""
        return self._call_llm(prompt, model=LLM_CONFIG["model_assessment"])


class PrayerVerifier:
    """Ensures generated prayers meet protocol requirements."""

    def __init__(self):
        """Defines the scar taxonomy for verification."""
        self.scar_categories = {
            "truncation": "Prayer body or Amen is cut off.",
            "span-incompleteness": "Scan text missing required F/M/L tags.",
            "seal-token-failure": "Assessment missing SEEDSEAL tokens.",
            "pattern-missing": "Fewer than the required (7/5/3) patterns used.",
            "formulaic": "Prayer body is repetitive or low-energy.",
            "shallow-reading": "Patterns are superficial, not structural.",
            "failed-verification": "Automation check failed (e.g., missing Amen).",
            "missing-witness": "WITNESS header or body mention is missing.",
            "missing-scar": "SCAR header is missing.",
            "boundary-violation": "LLM broke character or format.",
            "turbulent-flow": "Prayer structure is chaotic.",
            "anchor-shift": "Devotional anchor (Christ) is lost.",
            # ... (plus 52 other categories)
        }

    def _span_level_from_scan(self, scan_text: str) -> int:
        """Determines the 7/5/3 span level from the scan output text."""
        seven = all(tag in scan_text for tag in ["[FIRST1]","[FIRST2]","[FIRST3]","[MID1]","[MID2]","[LAST2]","[LAST1]"])
        five  = all(tag in scan_text for tag in ["[FIRST1]","[FIRST2]","[MID]","[LAST2]","[LAST1]"])
        three = all(tag in scan_text for tag in ["[FIRST]","[MID]","[LAST]"])
        return 7 if seven else 5 if five else 3 if three else 0

    def verify_branch_prayer(self, prayer_text, prayer_num):
        """Verify Branch prayer has exact format and detect automation scars."""
        checks = {}
        detected_scars = []

        title_pattern = rf"^PRAYER {prayer_num}\s*$"
        checks["has_correct_title"] = bool(re.search(title_pattern, prayer_text, re.M))
        if not checks["has_correct_title"]:
            detected_scars.append("failed-verification: has_correct_title")

        header_checks = {
            "has_checksum": re.search(r"^CHECKSUM:", prayer_text, re.M),
            "has_inheritance": re.search(r"^INHERITANCE:", prayer_text, re.M),
            "has_witness_field": re.search(r"^WITNESS:", prayer_text, re.M),
            "has_scar": re.search(r"^SCAR:", prayer_text, re.M),
        }
        for name, check in header_checks.items():
            checks[name] = bool(check)
            if not checks[name]:
                detected_scars.append(f"failed-verification: {name}")

        checks["has_prayer_label"] = bool(re.search(r"^Prayer\s*$", prayer_text, re.M))
        if not checks["has_prayer_label"]:
            detected_scars.append("failed-verification: has_prayer_label")

        checks["has_christ_witness"] = bool(re.search(r"\b(Christ|Lord|Jesus)\b", prayer_text))
        if not checks["has_christ_witness"]:
            detected_scars.append("missing-witness: body_mention")

        checks["has_amen"] = bool(re.search(r'(?im)\bamen\b\.?\s*$', prayer_text.strip()))
        if not checks["has_amen"]:
            detected_scars.append("truncation: missing_amen")

        passed = all(checks.values())
        if not passed:
            failed_checks = [k for k, v in checks.items() if not v]
            if DEBUG: print(_c("yellow", f"Verification Failed. Missing: {failed_checks}"))
        
        return passed, checks, detected_scars


class AsyncHumanAudit:
    """Manages the non-blocking human-in-the-loop audit process."""
    def __init__(self, base_path, orchestrator_config):
        self.audit_path = Path(base_path) / "audits"
        self.audit_path.mkdir(exist_ok=True)
        self.config = orchestrator_config
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_audits = {}

    def needs_audit(self, prayer_num):
        if prayer_num == 60 or prayer_num == 72:
            return True
        return False

    def _audit_worker(self, prayer_num, prayer_text, assessment_text):
        pending_file = self.audit_path / f"prayer_{prayer_num}_pending.txt"
        approved_file = self.audit_path / f"prayer_{prayer_num}_approved.txt"
        
        with open(pending_file, 'w', encoding='utf-8') as f:
            f.write(f"--- PRAYER {prayer_num} ---\n\n{prayer_text}\n\n" +
                    "="*20 + " ASSESSMENT " + "="*20 + "\n\n" +
                    assessment_text)
        
        print(f"\nAUDIT for Prayer {prayer_num} submitted. Review: {pending_file}")
        timeout_minutes = AUDIT_CONFIG.get('audit_timeout_minutes', 120)
        deadline = time.time() + (timeout_minutes * 60)
        
        while not approved_file.exists():
            if time.time() > deadline:
                print(f"AUDIT TIMEOUT for Prayer {prayer_num}. Continuing.")
                return False
            time.sleep(60)
            
        print(f"Audit for Prayer {prayer_num} approved.")
        return True

    def request_audit_async(self, prayer_num, prayer_text, assessment_text):
        if (self.audit_path / f"prayer_{prayer_num}_approved.txt").exists():
            print(f"Prayer {prayer_num} already approved.")
            return

        future = self.executor.submit(self._audit_worker, prayer_num, prayer_text, assessment_text)
        self.pending_audits[prayer_num] = future

    def check_pending_audits(self):
        completed = [p_num for p_num, future in self.pending_audits.items() if future.done()]
        for p_num in completed:
            del self.pending_audits[p_num]
        return sorted(self.pending_audits.keys())


class PrayerAutomation:
    """Manages all file I/O and protocol-related filesystem operations."""
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.paths = {
            "trunk": self.base_path / "Trunk",
            "branch": self.base_path / "Branch",
            "branch_assessments": self.base_path / "Branch_Assessments",
            "corpus": self.base_path / "Corpus",
            "log": self.base_path / "supervisor_log.txt",
            "scars_dir": self.base_path / "SCARS",
            "scars_ledger": self.base_path / "SCARS" / "llm_scars.jsonl",
            "scars_txt": self.base_path / "SCARS" / "scar_log.txt",
        }
        for p in self.paths.values():
            if isinstance(p, Path) and p.suffix == "":
                p.mkdir(parents=True, exist_ok=True)

    def get_prayer_path(self, prayer_num):
        """Get the path for a prayer file in the new structure."""
        if prayer_num <= 48:
            section = "trunk"
            scroll_num = ((prayer_num - 1) // 12) + 1
        else:
            section = "branch"
            scroll_num = ((prayer_num - 49) // 12) + 1
            
        scroll_path = self.paths[section] / f"Scroll_{scroll_num:02d}"
        scroll_path.mkdir(parents=True, exist_ok=True)
        return scroll_path / f"Prayer_{prayer_num:02d}.txt"

    def get_assessment_path(self, prayer_num):
        """Get the path for an assessment file (Branch only)."""
        if prayer_num < 49:
            raise ValueError("Assessments only for Branch prayers (49+).")
        scroll_num = ((prayer_num - 49) // 12) + 1
        scroll_path = self.paths["branch_assessments"] / f"Scroll_{scroll_num:02d}"
        scroll_path.mkdir(parents=True, exist_ok=True)
        return scroll_path / f"Prayer_{prayer_num:02d}_assessment.txt"

    def get_corpus_chunk(self, chunk_num):
        """Read a corpus chunk file."""
        try:
            corpus_file = next(self.paths["corpus"].glob(f"XX_{chunk_num:02d}*.txt"))
            with open(corpus_file, 'rb') as f:
                content = f.read()
            return content.decode('utf-8'), hashlib.sha256(content).hexdigest()
        except StopIteration:
            raise FileNotFoundError(f"No Corpus file found for chunk {chunk_num}")

    def read_file(self, path):
        if not path.exists():
            return ""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def read_prayer(self, prayer_num):
        if prayer_num < 1:
            return ""
        return self.read_file(self.get_prayer_path(prayer_num))

    def read_assessment(self, prayer_num):
        if prayer_num < 49:
            return ""
        return self.read_file(self.get_assessment_path(prayer_num))

    def save_prayer(self, prayer_num, prayer_text):
        with open(self.get_prayer_path(prayer_num), 'w', encoding='utf-8') as f:
            f.write(prayer_text)

    def save_assessment(self, prayer_num, assessment_text, automation_scars=None):
        path = self.get_assessment_path(prayer_num)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(assessment_text)
        self.extract_and_persist_scars(prayer_num, assessment_text, automation_scars)

    def extract_and_persist_scars(self, prayer_num, assessment_text, automation_scars=None):
        """Pull LLM SCARS and persist both automation and LLM scars to ledger."""
        now = datetime.datetime.now().isoformat()
        
        if automation_scars:
            for scar_desc in automation_scars:
                parts = scar_desc.split(":", 1)
                if len(parts) == 2:
                    cat, desc = parts[0].strip(), parts[1].strip()
                else:
                    cat, desc = "automation-failure", scar_desc
                
                record = {
                    "type": "automation-detected",
                    "prayer": prayer_num,
                    "timestamp": now,
                    "scar_category": cat,
                    "description": desc,
                }
                with open(self.paths["scars_ledger"], "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        m = re.search(r"LLM SCARS\s*\(CONFESSION\)\s*:\s*(.+?)(?:\n\s*\n|COVERAGE:|WITNESS:|Note for next reading:)", assessment_text, re.S | re.I)
        if not m:
            return

        block = m.group(1).strip()
        if not block or block.lower() == "[none]":
            return

        parsed_categories = set()
        raw_lines = block.splitlines()
        
        possible_tags = ["truncation", "pattern-missing", "formulaic", "shallow-reading", "failed-verification", "missing-witness", "missing-scar", "boundary-violation", "turbulent-flow", "anchor-shift", "span-incompleteness", "seal-token-failure"]
        
        for line in raw_lines:
            line = line.strip(" -*")
            if not line:
                continue
            for tag in possible_tags:
                if tag in line.lower():
                    parsed_categories.add(tag)
        
        record = {
            "type": "llm-confession",
            "prayer": prayer_num,
            "timestamp": now,
            "raw_scars": block,
            "parsed_categories": sorted(list(parsed_categories))
        }
        with open(self.paths["scars_ledger"], "a", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_to_supervisor(self, prayer_num, prayer_text):
        checksum = hashlib.sha256(prayer_text.encode('utf-8')).hexdigest()
        log_entry = (
            f"\n--- Prayer {prayer_num} Logged ---\n"
            f"Timestamp: {datetime.datetime.now().isoformat()}\n"
            f"Prayer Checksum: {checksum}\n"
        )
        with open(self.paths["log"], 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def add_scar(self, prayer_num, error_message):
        scar_entry = (
            f"\n!!! SCAR RECORDED: Prayer {prayer_num} Automation Failure !!!\n"
            f"Timestamp: {datetime.datetime.now().isoformat()}\n"
            f"Error: {error_message}\n"
        )
        with open(self.paths["log"], 'a', encoding='utf-8') as f:
            f.write(scar_entry)
        with open(self.paths["scars_txt"], 'a', encoding='utf-8') as f:
            f.write(scar_entry)
        
        record = {
            "type": "automation",
            "prayer": prayer_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "error": error_message,
        }
        with open(self.paths["scars_ledger"], "a", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")


class SeedOrchestrator:
    """Top-level class to run and manage the entire automation pipeline."""
    def __init__(self, api_key, config):
        self.config = config['seed_config']
        base_path = self.config['base_path']
        self.automation = PrayerAutomation(base_path)
        self.generator = PrayerGenerator(api_key, base_path)
        self.verifier = PrayerVerifier()
        self.auditor = AsyncHumanAudit(base_path, self.config)
        
        if DEBUG: print(_c("cyan", "[PREFLIGHT] Pinging OpenAI API..."))
        try:
            openai_api_wrapper.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=[{"role": "system", "content": "ping"}, {"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=1,
            )
            if DEBUG: print(_c("green", "[PREFLIGHT] OpenAI API ping successful."))
        except Exception as e:
            print(_c("red", f"[PREFLIGHT] OpenAI API ping FAILED."))
            raise RuntimeError(f"OpenAI preflight failed: {e}")
        
        self.current_prayer_num = 49
        self.load_state()

        print("Performing Corpus file preflight check...")
        corpus_dir = Path(base_path) / "Corpus"
        missing = []
        for i in range(1, 25):
            if not any(corpus_dir.glob(f"XX_{i:02d}*.txt")):
                missing.append(i)
        if missing:
            raise FileNotFoundError(f"CRITICAL: Missing Corpus files: {missing}")
        print("Corpus file check passed. All 24 chunks present.")

    def save_state(self):
        state = {'current_prayer_num': self.current_prayer_num}
        with open(Path(self.config['base_path']) / 'orchestrator_state.json', 'w') as f:
            json.dump(state, f)

    def load_state(self):
        state_file = Path(self.config['base_path']) / 'orchestrator_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            self.current_prayer_num = state.get('current_prayer_num', 49)
        print(f"State loaded. Resuming from Prayer {self.current_prayer_num}.")

    def _ensure_trailing_amen(self, text):
        """Ensure prayer ends with Amen."""
        if not re.search(r'(?im)\bamen\b\.?\s*$', text.strip()):
            return text.rstrip() + "\n\nAmen.", True
        return text, False

    def _ensure_witness_line(self, text: str):
        """Ensure WITNESS field has Christ/Lord mention and is placed correctly."""
        m = re.search(r"^WITNESS:\s*(.+)$", text, re.M)
        if m and re.search(r"\b(Christ|Lord|Jesus)\b", m.group(1)):
            return text, False

        new_witness = "WITNESS: Lord Jesus Christ is the Anchor and Witness of this reading"
        if m:
            # === 1. FIXED TYPO (WITTEST -> WITNESS) ===
            fixed = re.sub(r"^WITNESS:.*$", new_witness, text, flags=re.M)
            return fixed, True

        insert_after = re.search(r"^SCAR:.*$", text, re.M) or re.search(r"^INHERITANCE:.*$", text, re.M)
        if insert_after:
            i = insert_after.end()
            fixed = text[:i] + "\n" + new_witness + text[i:]
            return fixed, True

        prayer_header = re.search(r"^PRAYER\s+\d+\s*$", text, re.M)
        if prayer_header:
            i = prayer_header.end()
            fixed = text[:i] + "\n" + new_witness + text[i:]
            return fixed, True

        return new_witness + "\n" + text, True

    def _verify_coverage_seal(self, assessment_text: str, expected_tokens: list[str]) -> bool:
        """Verifies the COVERAGE SEAL in the assessment text."""
        m = re.search(r"(?im)^[\s#>\-\*]*COVERAGE\s*SEAL\s*:\s*(.+)$", assessment_text)
        if m:
            got = m.group(1).strip().split()
            return got == expected_tokens
        pos = -1
        for tok in expected_tokens:
            pos = assessment_text.find(tok, pos + 1)
            if pos == -1:
                return False
        return True

    def handle_branch_prayer(self, prayer_num):
        """Handle Branch prayers 49-72 with covenant format."""
        print(f"\n--- Starting Branch Prayer {prayer_num} ---")

        if prayer_num == 49:
            inheritance_str = "Trunk completion (Prayer 48) + Corpus XX_01"
            chunk_num = 1
        elif 50 <= prayer_num <= 60:
            inheritance_str = f"Prayer {prayer_num-1} + Corpus XX_{prayer_num-48:02d}"
            chunk_num = prayer_num - 48
        elif prayer_num == 61:
            inheritance_str = "Prayers 49–60 + Corpus XX_13"
            chunk_num = 13
        else:
            inheritance_str = f"Prayer {prayer_num-1} + Corpus XX_{prayer_num-48:02d}"
            chunk_num = prayer_num - 48

        corpus_content, corpus_hash = self.automation.get_corpus_chunk(chunk_num)
        
        if DEBUG: print(_c("cyan", f"[SCAN] Generating corpus scan for XX_{chunk_num:02d}"))
        scan_text, seal_tokens, coverage = self.generator.generate_corpus_scan(corpus_content)
        
        span_floor = self.verifier._span_level_from_scan(scan_text)
        if span_floor == 0:
            print(_c("red", "[SCAN] Failed: span-incompleteness. Scan missing F/M/L tags."))
            self.automation.add_scar(prayer_num, "span-incompleteness: missing span tags in scan")
        elif DEBUG:
            print(_c("green", f"[SCAN] Scan OK. Detected span floor: {span_floor}"))
            # === 2. ADDED SCAR FOR PARTIAL SPAN ===
            if span_floor in (5, 3):
                self.automation.add_scar(prayer_num, f"span-incompleteness: only {span_floor} span notes")

        checksum_desc = inheritance_str
        checksum_digest = hashlib.sha256(
            (checksum_desc + corpus_content).encode('utf-8')
        ).hexdigest()

        scar_line = self.generator._get_scar_line(last_n=1)

        prayer = self.generator.generate_branch_prayer(
            prayer_num=prayer_num,
            inheritance_str=inheritance_str,
            scar_line=scar_line,
            corpus_content=corpus_content,
            checksum_desc=checksum_desc,
            checksum_digest=checksum_digest,
            scan_text=scan_text,
            coverage=coverage,
            span_floor=span_floor
        )

        if f"{checksum_digest}" not in prayer:
            print(_c("red", "[VERIFY] Failed: Checksum digest not found in prayer text."))
            self.automation.add_scar(prayer_num, "failed-verification: checksum digest not found in CHECKSUM line")
        elif DEBUG:
            print(_c("green", "[VERIFY] Checksum digest OK (found in text)."))

        prayer, amen_fixed = self._ensure_trailing_amen(prayer)
        prayer, witness_fixed = self._ensure_witness_line(prayer)
        
        is_valid, checks, detected_scars = self.verifier.verify_branch_prayer(prayer, prayer_num)

        if amen_fixed:
            self.automation.add_scar(prayer_num, "auto-fix: appended terminal Amen.")
            if "truncation: missing_amen" not in detected_scars:
                 detected_scars.append("truncation: missing_amen (auto-fixed)")
            if DEBUG: print(_c("yellow", "[FIX] appended Amen"))
            
        if witness_fixed:
            self.automation.add_scar(prayer_num, "auto-fix: fixed WITNESS line.")
            if "failed-verification: has_witness_field" not in detected_scars:
                detected_scars.append("failed-verification: has_witness_field (auto-fixed)")
            if DEBUG: print(_c("yellow", "[FIX] fixed WITNESS line"))

        if not is_valid:
            print(_c("red", f"[VERIFY] failed for Prayer {prayer_num}"))
        else:
            if DEBUG: print(_c("green", "[VERIFY] ok"))

        self.automation.save_prayer(prayer_num, prayer)
        self.automation.log_to_supervisor(prayer_num, prayer)

        assessment = self.generator.generate_assessment(
            prayer_num=prayer_num,
            prayer_text=prayer,
            corpus_content=corpus_content,
            detected_scars=detected_scars,
            seal_tokens=seal_tokens,
            coverage=coverage
        )
        
        # === 5. ADDED SPAN LEVEL TO ASSESSMENT FILE ===
        assessment += f"\n\n[SPAN LEVEL]: {span_floor or 0}"
        
        self.automation.save_assessment(
            prayer_num, 
            assessment, 
            automation_scars=detected_scars
        )

        # === 3. ADDED CONDITIONAL SEAL CHECK ===
        if seal_tokens:
            ok_seal = self._verify_coverage_seal(assessment, seal_tokens)
            if not ok_seal:
                print(_c("red", "[VERIFY] Failed: seal-token-failure. Coverage seal mismatch."))
                self.automation.add_scar(prayer_num, "seal-token-failure: coverage seal mismatch")
            elif DEBUG:
                print(_c("green", "[VERIFY] Coverage Seal OK."))

        if DEBUG:
            print(_c("green", f"[SAVE] Branch Prayer {prayer_num} complete"))
            print(_c("cyan", f" Inheritance: {inheritance_str}"))
            print(_c("cyan", f" Corpus: XX_{chunk_num:02d}"))

        if self.auditor.needs_audit(prayer_num):
            self.auditor.request_audit_async(prayer_num, prayer, assessment)

    def run_continuous(self):
        """Run the Branch prayer generation loop (49-72)."""
        print(f"Starting Branch Generation. Beginning with Prayer {self.current_prayer_num}.")
        
        while self.current_prayer_num <= 72:
            pending = self.auditor.check_pending_audits()
            if pending and (self.current_prayer_num - min(pending)) >= AUDIT_CONFIG.get('max_pending_audits', 5):
                print(f"Pausing. Waiting for {len(pending)} pending audits: {pending}")
                time.sleep(self.config.get('rate_limit_delay', 1))
                continue

            prayer_num = self.current_prayer_num
            try:
                self.handle_branch_prayer(prayer_num)
                self.current_prayer_num += 1
                self.save_state()
                time.sleep(self.config.get('rate_limit_delay', 1))
                
            except Exception as e:
                print(f"\n!!! AUTOMATION HALTED on Prayer {prayer_num} !!!")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                self.automation.add_scar(prayer_num, str(e))
                self.save_state()
                print(f"Pausing for {self.config.get('error_pause_duration', 30)} seconds.")
                time.sleep(self.config.get('error_pause_duration', 30))

        print("\n=== Branch Generation Complete (Prayers 49-72) ===")


if __name__ == "__main__":
    if DEBUG:
        import sys
        from datetime import datetime as _dt
        tee_path = Path(CONFIG['seed_config']['base_path']) / f"branch_run_{_dt.now().strftime('%Y%m%d_%H%M%S')}.log"
        tee = open(tee_path, "w", encoding="utf-8")
        
        class _Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                tee.write(s)
            def flush(self):
                sys.__stdout__.flush()
                tee.flush()
        sys.stdout = _Tee()

    orchestrator = SeedOrchestrator(api_key=OPENAI_API_KEY, config=CONFIG)
    orchestrator.run_continuous()