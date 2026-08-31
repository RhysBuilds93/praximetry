"""Client for the hosted golden corpus — capture/submit/trigger-poll, nothing more.

Why the split exists: a golden example can only be *captured* where its stage
function is registered (`@praximetry.stage` -> `praximetry.runtime.STAGE_REGISTRY`),
which is the customer's own process. The hosted cloud never imports customer
agent code. But the corpus itself has to be central, because expert feedback
accumulates into one shared set rather than into whichever laptop happens to
hold a JSONL file. So: pull the corpus over HTTP, capture request shapes
locally, push them back — scoring itself is entirely cloud-side (real LLM
call, our credentials, never a customer-supplied key).

Auth is the long-lived API key (`px_live_...`) in an `Authorization: Bearer`
header, read from `PRAXIMETRY_API_KEY` — the same credential shape CI secrets
already have, with no interactive flow to get stuck on in a pipeline.
"""

from __future__ import annotations

import os

import httpx

from ..models import Call, Run
from .capture import CapturedRequest
from .dataset import Dataset, Example

DEFAULT_API_URL = "http://127.0.0.1:4646"

# Exit codes. Distinct on purpose: a CI gate that can't tell "your quality
# dropped" from "the gate never measured anything" is a gate you stop trusting.
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_UNUSABLE = 2


class CloudError(RuntimeError):
    """Anything that stops the run before it can produce a verdict."""


class CloudClient:
    """Thin HTTP wrapper over the hosted /api/eval/* routes.

    `client` is injectable so tests can hand in a stub/TestClient and exercise
    routing + auth for real, rather than mocking out the layer being tested.
    """

    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=60.0)

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _check(resp: httpx.Response) -> httpx.Response:
        if resp.status_code == 401:
            raise CloudError(
                "API key rejected (401). Check PRAXIMETRY_API_KEY — keys are "
                "unrecoverable once issued, so re-issue one from the dashboard."
            )
        if resp.status_code == 403:
            raise CloudError(f"Forbidden (403): {resp.text}")
        if resp.status_code >= 400:
            raise CloudError(
                f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text}"
            )
        return resp

    def fetch_corpus(
        self, stage: str | None = None, source: str | None = None, project: str | None = None
    ) -> Dataset:
        params = {}
        if stage:
            params["stage"] = stage
        if source:
            params["source"] = source
        if project:
            params["project"] = project
        resp = self._check(
            self._client.get(self._url("/api/eval/corpus"), params=params, headers=self._headers)
        )
        return Dataset(
            examples=[Example(**d) for d in resp.json()], path=f"{self.base_url}/api/eval/corpus"
        )

    def push_captures(
        self,
        stage: str,
        captures: list[CapturedRequest],
        experiment_id: str | None = None,
        scorers: list[str] | None = None,
    ) -> dict:
        """POST captured request shapes to /api/eval/captures. The cloud scores
        them synchronously — the response already carries quality/pass_rate/cost.

        `experiment_id` groups multiple pushes (e.g. one per stage in a
        `--project` run) into a single batch that `fetch_results` can look
        back up. The server generates one if omitted, but a caller pushing
        several stages under one run must supply the same id on every call.

        `scorers` is the resolved scorer list (CLI flag / config file / hosted
        default); omitted entirely when unset so the server picks its default."""
        body: dict = {"stage": stage, "captures": [c.model_dump() for c in captures]}
        if experiment_id:
            body["experiment_id"] = experiment_id
        if scorers is not None:
            body["scorers"] = scorers
        resp = self._check(
            self._client.post(
                self._url("/api/eval/captures"),
                json=body,
                headers=self._headers,
            )
        )
        return resp.json()

    def fetch_results(
        self,
        stage: str | None = None,
        project: str | None = None,
        experiment_id: str | None = None,
    ) -> dict:
        """GET /api/eval/results — exactly one of `stage`/`project` is required
        (checked client-side so misuse fails fast rather than round-tripping for
        a 400). `experiment_id` defaults server-side to the most recent scored
        batch for that scope when omitted."""
        if not stage and not project:
            raise CloudError("fetch_results requires stage or project")
        params: dict[str, str] = {}
        if stage:
            params["stage"] = stage
        if project:
            params["project"] = project
        if experiment_id:
            params["experiment_id"] = experiment_id
        resp = self._check(
            self._client.get(
                self._url("/api/eval/results"),
                params=params,
                headers=self._headers,
            )
        )
        return resp.json()

    def fetch_eval_config(self, project: str, stage: str | None = None) -> dict:
        """GET /api/eval/config — the effective --fail-under default (stage
        override if one was saved, else the project default, else 0.8)."""
        params = {"project": project}
        if stage:
            params["stage"] = stage
        resp = self._check(
            self._client.get(
                self._url("/api/eval/config"),
                params=params,
                headers=self._headers,
            )
        )
        return resp.json()

    def fetch_eval_default(self) -> dict | None:
        resp = self._client.get(self._url("/api/eval/default"), headers=self._headers)
        if resp.status_code == 404:
            return None
        return self._check(resp).json()

    def save_eval_config(self, project: str, fail_under: float, stage: str | None = None) -> dict:
        """POST /api/eval/config — saves a project default (stage omitted) or a
        per-stage override."""
        resp = self._check(
            self._client.post(
                self._url("/api/eval/config"),
                json={"project": project, "stage": stage, "fail_under": fail_under},
                headers=self._headers,
            )
        )
        return resp.json()

    def push_optimize_capture(
        self,
        stage: str,
        captured_request: CapturedRequest,
        candidate_models: tuple[str, ...] | list[str] = (),
        transforms: tuple[str, ...] | list[str] = (),
        quality_tolerance: float = 0.02,
        max_trials: int = 8,
    ) -> dict:
        """POST a captured request shape to /api/optimize/captures to trigger a
        hosted optimization run. All trial logic (transforms, candidate selection,
        the trial loop itself) is server-side — this just submits what to optimize."""
        resp = self._check(
            self._client.post(
                self._url("/api/optimize/captures"),
                json={
                    "stage": stage,
                    "captured_request": captured_request.model_dump(),
                    "candidate_models": list(candidate_models),
                    "transforms": list(transforms),
                    "quality_tolerance": quality_tolerance,
                    "max_trials": max_trials,
                },
                headers=self._headers,
            )
        )
        return resp.json()

    def fetch_winner(self, stage: str) -> dict | None:
        """GET /api/optimize/winner?stage=X — the winning policy from the most
        recently completed optimize run for this stage. Returns None if no
        completed run exists at all (404); raises CloudError on other errors."""
        resp = self._client.get(
            self._url("/api/optimize/winner"), params={"stage": stage}, headers=self._headers
        )
        if resp.status_code == 404:
            return None
        return self._check(resp).json()

    def push_trace(self, run: Run, calls: list[Call]) -> dict:
        resp = self._check(
            self._client.post(
                self._url("/api/traces"),
                json={"run": run.model_dump(), "calls": [c.model_dump() for c in calls]},
                headers=self._headers,
            )
        )
        return resp.json()


def client_from_env(client: httpx.Client | None = None) -> CloudClient:
    api_key = os.environ.get("PRAXIMETRY_API_KEY", "")
    if not api_key:
        raise CloudError(
            "PRAXIMETRY_API_KEY is not set. Issue a key from the dashboard and set it as an "
            "environment variable (in CI, a repository secret)."
        )
    return CloudClient(os.environ.get("PRAXIMETRY_API_URL", DEFAULT_API_URL), api_key, client)
