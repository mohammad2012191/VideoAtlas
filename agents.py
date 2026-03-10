"""
agents.py — Master and Worker VLM agents.

Supports two backends, configured via config.BACKEND:
  - "google"  → Google AI (Gemini) API using an API key
  - "vertex"  → Vertex AI using a GCP service account JSON

MasterAgent handles: sufficiency checks, uncertainty analysis, final decisions.
WorkerAgent handles: regional exploration.
"""

import io
import re
import json
import time
import base64
import random

from openai import OpenAI
from PIL import Image

from config import (
    BACKEND,
    MASTER_MODEL_PATH, WORKER_MODEL_PATH,
    MAX_PROBE_RETRIES, GRID_K,
    # Google AI settings
    GOOGLE_API_KEY, GOOGLE_BASE_URL,
    # Vertex AI settings
    VERTEX_BASE_URL, SERVICE_ACCOUNT_FILE,
)
from metrics import metrics
from memory import _idx_to_letter, _letter_to_idx
from logger import log


# ==========================================
# TOKEN / CLIENT HELPERS
# ==========================================
def _get_vertex_token():
    """Obtain a short-lived OAuth2 token for Vertex AI."""
    import google.oauth2.service_account
    import google.auth.transport.requests
    creds = google.oauth2.service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _build_client() -> OpenAI:
    """
    Return an OpenAI-compatible client pointed at the configured backend.

    - "google" : Google AI (Gemini) OpenAI-compatible endpoint, authenticated
                 with a static API key.
    - "vertex" : Vertex AI OpenAI-compatible endpoint, authenticated with a
                 short-lived OAuth2 token derived from a service-account file.
    """
    if BACKEND == "google":
        return OpenAI(
            base_url=GOOGLE_BASE_URL,
            api_key=GOOGLE_API_KEY,
            timeout=120.0,
        )
    else:  # "vertex"
        return OpenAI(
            base_url=VERTEX_BASE_URL,
            api_key=_get_vertex_token(),
            timeout=120.0,
        )


# ==========================================
# BASE AGENT
# ==========================================
class _BaseAgent:
    def __init__(self, model_path: str, gpu_id: int, is_master: bool):
        self.gpu_id     = gpu_id
        self.is_master  = is_master
        self.model_name = model_path
        role    = "MASTER" if is_master else "WORKER"
        backend = BACKEND.upper()
        print(f"[{role} GPU {gpu_id}] Backend={backend}  model={model_path}")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pil_to_b64(img) -> str:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------
    def _to_oai_messages(self, messages):
        oai = []
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                oai.append({"role": role, "content": content})
            else:
                parts = []
                for part in content:
                    if part["type"] == "image":
                        b64 = self._pil_to_b64(part["image"])
                        parts.append({
                            "type":      "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
                    else:
                        parts.append(part)
                oai.append({"role": role, "content": parts})
        return oai

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def _generate(self, messages, tools=None, max_tokens: int = 2048,
                  thinking: bool = False) -> str:
        oai_messages = self._to_oai_messages(messages)
        kwargs = dict(
            model=self.model_name,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
        )
        if tools:
            kwargs["tools"]       = tools
            kwargs["tool_choice"] = "required"

        for attempt in range(6):
            try:
                client   = _build_client()          # re-built each call (token may expire)
                response = client.chat.completions.create(**kwargs)
                usage    = response.usage
                if usage:
                    metrics.add_call(usage.prompt_tokens, usage.completion_tokens,
                                     is_master=self.is_master)

                if not response.choices:
                    raise ValueError("Empty choices in response")

                msg_out = response.choices[0].message
                if msg_out is None:
                    raise ValueError("response.choices[0].message is None")

                if msg_out.tool_calls:
                    parts = []
                    for tc in msg_out.tool_calls:
                        parts.append(
                            f'<tool_call>{{"name": "{tc.function.name}", '
                            f'"arguments": {tc.function.arguments}}}</tool_call>'
                        )
                    return "\n".join(parts)

                return msg_out.content or ""

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[429] Attempt {attempt+1}/6, retrying in {wait:.1f}s")
                    time.sleep(wait)
                elif "NoneType" in err or "choices" in err or "message is None" in err:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[None response] Attempt {attempt+1}/6, retrying in {wait:.1f}s — {e}")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_single_tc(self, json_str: str) -> dict:
        parsed    = json.loads(json_str)
        action    = parsed.get("name", parsed.get("action", ""))
        arguments = parsed.get("arguments", {})
        if isinstance(arguments, list):
            arguments = {"items": arguments} if action == "ADD_TO_SCRATCHPAD" else {"values": arguments}
        result = {"action": action}
        if isinstance(arguments, dict):
            result.update(arguments)
        return result

    def _parse_tool_calls(self, output_text: str) -> list:
        clean = output_text.strip()
        if "<tool_call>" in clean:
            matches = re.findall(r'<tool_call>(.*?)</tool_call>', clean, re.DOTALL)
            if matches:
                return [self._parse_single_tc(m.strip()) for m in matches]
        if "{" in clean and "}" in clean:
            start, end = clean.find('{'), clean.rfind('}') + 1
            return [self._parse_single_tc(clean[start:end])]
        raise ValueError("No tool call found")

    def _parse_json(self, output_text: str) -> dict:
        clean = output_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            parts = clean.split("```")
            if len(parts) >= 2:
                clean = parts[1]
        clean = clean.strip()
        start = clean.find('{')
        end   = clean.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in output: {output_text[:200]!r}")
        return json.loads(clean[start:end])


# ==========================================
# MASTER AGENT
# ==========================================
class MasterAgent(_BaseAgent):
    def __init__(self, gpu_id: int = 0):
        super().__init__(MASTER_MODEL_PATH, gpu_id, is_master=True)

    def check_sufficiency(self, query: str, scratchpad) -> str:
        evidence_img, cell_descs = scratchpad.generate_evidence_grid(cell_size=256)
        evidence_text = "\n".join(cell_descs)

        prompt = f"""**EVIDENCE SUFFICIENCY CHECK**

The grid image shows the most relevant frames collected so far from a SINGLE video, in chronological order (left-to-right, top-to-bottom).

**QUERY:** "{query}"

**EVIDENCE (chronologically ordered frames from the video):**
{evidence_text}

Has the search task been completed? Reply with ONLY "yes" or "no"."""

        messages = [{"role": "user", "content": [
            {"type": "image", "image": evidence_img},
            {"type": "text",  "text":  prompt}
        ]}]
        output = self._generate(messages, tools=None, max_tokens=2048, thinking=True)
        return "yes" if "yes" in output.lower() else "no"

    def final_decide(self, query: str, scratchpad, candidates=None) -> dict:
        evidence_img, cell_descs = scratchpad.generate_evidence_grid(cell_size=256)
        evidence_text = "\n".join(cell_descs)
        last_letter   = _idx_to_letter(len(cell_descs) - 1)

        if candidates:
            # ── Multiple-choice mode ──────────────────────────────────
            candidates_str = "\n".join([f"  {i}: {c}" for i, c in enumerate(candidates)])
            prompt = f"""**FINAL DECISION**

You are answering a question about a SINGLE video.

**QUERY:** "{query}"

**ANSWER CHOICES:**
{candidates_str}

**EVIDENCE (sorted by timestamp — [A] is earliest, [{last_letter}] is latest):**
{evidence_text}

**CRITICAL: Use ONLY the evidence above. Do NOT use your own knowledge.**

**STEP-BY-STEP:**
1. Review the search task results: what did the evidence find?
2. For EACH candidate (0 to {len(candidates)-1}):
   - Which evidence items [by letter] support it?
   - Which contradict it?
   - Verdict: SUPPORTED, UNSUPPORTED, or CONTRADICTED
3. Pick the SUPPORTED candidate with the strongest and most direct video evidence.

**OUTPUT (raw JSON — reason FIRST, then answer):**
{{"reasoning": "<step-by-step reasoning>", "answer": "<your answer>", "choice": <index>}}"""

        else:
            # ── Open-ended mode ───────────────────────────────────────
            prompt = f"""**FINAL ANSWER**

You are answering a question about a SINGLE video.

**QUERY:** "{query}"

**EVIDENCE (sorted by timestamp — [A] is earliest, [{last_letter}] is latest):**
{evidence_text}

**CRITICAL: Base your answer ONLY on the evidence above. Do NOT use your own knowledge.**

**STEP-BY-STEP:**
1. Review what each evidence item shows and what it tells you.
2. Synthesise the evidence into a clear, direct answer to the query.
3. If the evidence is insufficient, say so and explain what was found.

**OUTPUT (raw JSON — reason FIRST, then answer):**
{{"reasoning": "<step-by-step reasoning>", "answer": "<your complete answer>"}}"""

        messages = [{"role": "user", "content": [
            {"type": "image", "image": evidence_img},
            {"type": "text",  "text":  prompt}
        ]}]
        output = self._generate(messages, tools=None, max_tokens=8096, thinking=True)

        result = None
        try:
            result = self._parse_json(output)
        except Exception as e:
            log(f"[MASTER] final_decide parse failed: {e}, retrying...")

        for attempt in range(MAX_PROBE_RETRIES):
            if result is not None:
                break
            fix_prompt = f"""Fix this malformed JSON and return ONLY valid JSON.

Expected: {{"reasoning": "<str>", "answer": "<str>", "choice": <int or omit if open-ended>}}

Broken output:
{output[:3000]}

**OUTPUT (valid JSON only):**"""
            fix_msgs = [{"role": "user", "content": fix_prompt}]
            fixed    = self._generate(fix_msgs, tools=None, max_tokens=2048, thinking=True)
            try:
                result = self._parse_json(fixed)
                log(f"[MASTER] final_decide JSON repair attempt {attempt+1} succeeded")
            except Exception as e2:
                log(f"[MASTER] final_decide repair attempt {attempt+1}/{MAX_PROBE_RETRIES} failed: {e2}")
                output = fixed

        if result is None:
            log("[MASTER] final_decide: all retries failed, returning raw answer text")
            result = {"reasoning": "JSON parse failed after retries", "answer": output.strip()}

        result.setdefault("choice", -1)
        return result

    def uncertainty_analysis(self, query: str, scratchpad, candidates, grid_img,
                              cell_info, progress_text: str, num_suggestions: int) -> dict:
        from navigator import build_context_str
        evidence_img, cell_descs = scratchpad.generate_evidence_grid(cell_size=256)
        evidence_text = "\n".join(cell_descs)
        context_str   = build_context_str(cell_info)

        if candidates:
            # ── Multiple-choice mode ──────────────────────────────────
            candidates_str    = "\n".join([f"  {i}: {c}" for i, c in enumerate(candidates)])
            task_section      = f"**ANSWER CHOICES:**\n{candidates_str}"
            uncertainty_task  = "1. **UNCERTAINTY CHECK:** Which choices still lack sufficient evidence?"
            finish_condition  = "**If ALL choices are covered, output:**"
        else:
            # ── Open-ended mode ───────────────────────────────────────
            task_section      = "*(Open-ended question — no fixed answer choices)*"
            uncertainty_task  = "1. **UNCERTAINTY CHECK:** Does the evidence collected so far fully answer the query, or are key parts still missing?"
            finish_condition  = "**If the evidence is sufficient to give a complete answer, output:**"

        prompt = f"""**UNCERTAINTY ANALYSIS**

**QUERY:** "{query}"

{task_section}

**EVIDENCE COLLECTED SO FAR:**
{evidence_text}

**EXPLORATION PROGRESS:**
{progress_text}

**NAVIGATION GRID (blacked-out cells = already explored):**
{context_str}

**YOUR 3 TASKS:**
{uncertainty_task}
2. **EXPLORE SUGGESTIONS:** Suggest up to {num_suggestions} regions to explore next (non-blacked cells only).
3. **ERASE NOISE:** List evidence letters to remove if completely unrelated to the query.

{finish_condition}
{{"reasoning": "<why>", "action": "FINAL_DECISION"}}

**Otherwise output:**
{{"reasoning": "<what's missing>", "action": "CONTINUE", "explore": [<cell_id ints OR {{"start": <float>, "end": <float>}}>], "erase": [<evidence letters>]}}

**OUTPUT (raw JSON only):**"""

        content_parts = [{"type": "text", "text": prompt}]
        if scratchpad.evidence:
            content_parts.insert(0, {"type": "image", "image": evidence_img})
        if grid_img is not None:
            content_parts.insert(0, {"type": "image", "image": grid_img})

        messages = [{"role": "user", "content": content_parts}]
        output   = self._generate(messages, tools=None, max_tokens=2048)
        log(f"[MASTER] Uncertainty raw: {output[:300]}")

        result = None
        try:
            result = self._parse_json(output)
        except Exception as e:
            log(f"[MASTER] Uncertainty parse failed: {e}, retrying...")

        for attempt in range(MAX_PROBE_RETRIES):
            if result is not None:
                break
            fix_prompt = f"""Fix this malformed JSON and return ONLY valid JSON.

Expected: {{"reasoning": "<str>", "action": "CONTINUE"|"FINAL_DECISION", "explore": [...], "erase": [...]}}

Broken:
{output[:3000]}

**OUTPUT (valid JSON only):**"""
            fix_msgs = [{"role": "user", "content": fix_prompt}]
            fixed    = self._generate(fix_msgs, tools=None, max_tokens=2048, thinking=True)
            try:
                result = self._parse_json(fixed)
                log(f"[MASTER] Uncertainty JSON repair attempt {attempt+1} succeeded")
            except Exception as e2:
                log(f"[MASTER] Repair attempt {attempt+1}/{MAX_PROBE_RETRIES} failed: {e2}")
                output = fixed

        if result is None:
            log("[MASTER] All retries failed, defaulting to CONTINUE")
            result = {"action": "CONTINUE", "reasoning": "parse error", "explore": [], "erase": []}

        result["action"] = (
            "FINAL_DECISION"
            if result.get("action", "").upper() == "FINAL_DECISION"
            else "CONTINUE"
        )
        result.setdefault("explore", [])
        result.setdefault("erase", [])
        result.setdefault("reasoning", "")
        return result


# ==========================================
# WORKER AGENT
# ==========================================
class WorkerAgent(_BaseAgent):
    def __init__(self, gpu_id: int = 0):
        super().__init__(WORKER_MODEL_PATH, gpu_id, is_master=False)