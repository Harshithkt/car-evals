"""Optional "second opinion" from a vision LLM on Nebius Token Factory (OpenAI-compatible API).

It is not part of the measured recognition pipeline. It exists for cars the trained models do not know (for example
India-only models such as Maruti Suzuki, Tata or Mahindra): the user clicks a button, the vehicle crop is sent to the
LLM, and its free-text answer is shown next to the classifier's result, clearly labelled as unevaluated.

Configuration (environment variables, or a git-ignored `.env` file in the project root):
    NEBIUS_API_KEY     enables the feature (never sent to the browser, never logged)
    NEBIUS_BASE_URL    default https://api.tokenfactory.nebius.com/v1
    NEBIUS_VLM_MODEL   model id; default: picked from the key's /models list (a Qwen VL model if available)
    LLM_MAX_PER_HOUR   calls allowed per hour on this server, default 60 (protects the API credits on a public link)
"""
from __future__ import annotations

import base64
import collections
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# Chosen by a bake-off on Nebius (Nov 2026, details in docs/llm_bakeoff.md): 14 of the key's 30 models accept images; the 8
# best were run on 30 Indian-market cars from Wikimedia Commons with this exact prompt. muse-spark-open named all 30 (exact
# model) and also 18/18 US and Stanford cars, median 3.6 s; DeepSeek-V4.1-Flash 26/30, MiniMax-M3 24/30, gemma-3-27b-it 22/30.
PREFERRED = ('muse-spark', 'gemma-3-27b', 'gemma-3', 'qwen2.5-vl-72b', 'qwen2.5-vl', 'qwen2-vl-72b', 'qwen3-vl', '-vl-', 'vl', 'vision')
PROMPT = (
    'You are a car expert. Identify the car in this photo as precisely as you can. Consider cars sold anywhere in the world, '
    'including India-only models (for example Maruti Suzuki, Tata, Mahindra), China-only and Europe-only models. '
    'Reply with JSON only, no other text, in exactly this form: '
    '{"make": "...", "model": "...", "generation_or_years": "...", "body_type": "...", '
    '"confidence": "high|medium|low", "clues": "one short sentence naming the visual clues you used"}. '
    'If you cannot tell, use "unknown" for the fields you are unsure of and set confidence to "low".'
)


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from ROOT/.env without overriding real environment variables (no extra dependency)."""
    p = ROOT / '.env'
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class LLMError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class VisionLLM:
    def __init__(self) -> None:
        self.key = os.environ.get('NEBIUS_API_KEY', '').strip()
        self.base = os.environ.get('NEBIUS_BASE_URL', 'https://api.tokenfactory.nebius.com/v1').rstrip('/')
        self.model = os.environ.get('NEBIUS_VLM_MODEL', '').strip() or None
        self.max_per_hour = int(os.environ.get('LLM_MAX_PER_HOUR', '60'))
        self.calls: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _request(self, path: str, payload: dict | None = None, timeout: float = 55) -> dict:
        req = urllib.request.Request(
            self.base + path, method='POST' if payload is not None else 'GET',
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json', 'User-Agent': 'p5-vehicle-recognition'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors='replace')
            if e.code in (401, 403):
                raise LLMError(502, 'The LLM provider rejected the API key.') from None
            raise LLMError(502, f'The LLM provider returned HTTP {e.code}: {body}') from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(504, f'Could not reach the LLM provider ({type(e).__name__}).') from None

    def pick_model(self) -> str:
        """Configured model, else the best vision model in the key's /models list."""
        if self.model:
            return self.model
        ids = [m.get('id', '') for m in self._request('/models', timeout=20).get('data', [])]
        for pref in PREFERRED:
            hit = sorted((i for i in ids if pref in i.lower()), key=lambda i: ('instruct' not in i.lower(), len(i)))
            if hit:
                self.model = hit[0]
                return self.model
        raise LLMError(502, 'No vision model found for this API key. Set NEBIUS_VLM_MODEL to a model that accepts images.')

    def _take_slot(self) -> None:
        now = time.time()
        with self.lock:
            while self.calls and now - self.calls[0] > 3600:
                self.calls.popleft()
            if len(self.calls) >= self.max_per_hour:
                wait = int(3600 - (now - self.calls[0])) // 60 + 1
                raise LLMError(429, f'The LLM second opinion is limited to {self.max_per_hour} calls per hour on this server. Try again in about {wait} min.')
            self.calls.append(now)

    def identify(self, img: Image.Image) -> dict:
        if not self.available:
            raise LLMError(503, 'No LLM is configured on this server (NEBIUS_API_KEY is not set).')
        self._take_slot()
        im = img.convert('RGB')
        im.thumbnail((768, 768), Image.LANCZOS)          # enough detail for badges and lights, few image tokens
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=88)
        model = self.pick_model()
        t0 = time.time()
        out = self._request('/chat/completions', {
            'model': model, 'temperature': 0.1, 'max_tokens': 1500,   # room for models that reason before answering
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()}},
            ]}],
        })
        text = (out.get('choices') or [{}])[0].get('message', {}).get('content') or ''
        if not text.strip():
            raise LLMError(502, f'{model} returned an empty answer. Set NEBIUS_VLM_MODEL to a vision model such as google/gemma-3-27b-it.')
        answer = None
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            try:
                answer = json.loads(m.group(0))
            except json.JSONDecodeError:
                answer = None
        if isinstance(answer, dict):
            answer = {k: str(answer.get(k, '')).strip()[:200] for k in ('make', 'model', 'generation_or_years', 'body_type', 'confidence', 'clues')}
        return {'model': model, 'answer': answer, 'text': None if answer else text.strip()[:600],
                'latency_ms': round((time.time() - t0) * 1000), 'usage': out.get('usage')}


LLM = VisionLLM()
