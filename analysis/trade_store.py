"""Saved trading data (open positions, closed trades and pre-trade settings) that outlives an app restart.

Streamlit Cloud wipes the app's disk on every reboot or redeploy, so the data is kept as one JSON file in a
private GitHub repository through the GitHub contents API (``GitHubStore``). Each save is a commit, so the
repository also keeps a full history of changes. Without a token the data goes to a local file
(``LocalStore``), which lasts only until the app restarts.

Streamlit secrets for the GitHub store::

    [trade_store]
    github_token = "github_pat_..."   # fine-grained token: this one repository, Contents read and write
    repo = "owner/backtester-data"     # a private repository
    path = "trading.json"              # optional
"""

import base64
import copy
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

EMPTY_DATA = {
    "version": 1,
    "settings": {"account": 100_000.0, "slot_pct": 4.0, "stop_pct": 5.0, "max_positions": 15},
    "positions": [],
    "closed": [],
}
POSITION_FIELDS = ["Symbol", "Shares", "Entry price", "Entry date", "Stop", "Note"]
CLOSED_FIELDS = ["Symbol", "Shares", "Entry price", "Entry date", "Exit price", "Exit date", "P&L", "Note"]


class StoreError(Exception):
    """The store could not be read or written; the message is shown to the user."""


class StoreConflict(StoreError):
    """The saved file changed since it was read (another device saved first)."""


def empty_data():
    return copy.deepcopy(EMPTY_DATA)


def normalize(data):
    """Fill in anything missing from older or hand-edited files."""

    result = empty_data()
    if isinstance(data, dict):
        result["settings"].update({k: v for k, v in (data.get("settings") or {}).items() if v is not None})
        result["positions"] = [row for row in data.get("positions") or [] if isinstance(row, dict)]
        result["closed"] = [row for row in data.get("closed") or [] if isinstance(row, dict)]
    return result


class LocalStore:
    """A JSON file on the app's own disk. Kept across page reloads, lost when Streamlit Cloud restarts the app."""

    durable = False

    def __init__(self, path):
        self.path = Path(path)

    @property
    def label(self):
        return "this app's own disk (lost when the app restarts)"

    def load(self):
        try:
            return normalize(json.loads(self.path.read_text())), None
        except FileNotFoundError:
            return empty_data(), None
        except (OSError, ValueError) as error:
            raise StoreError(f"Could not read {self.path}: {error}") from error

    def save(self, data, version=None):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1, default=str))
        except OSError as error:
            raise StoreError(f"Could not write {self.path}: {error}") from error
        return None


class GitHubStore:
    """One JSON file in a GitHub repository. ``version`` is the file's blob SHA, used to refuse stale saves."""

    durable = True
    API = "https://api.github.com"

    def __init__(self, token, repo, path="trading.json", timeout=20):
        self.token, self.repo, self.path, self.timeout = token, repo.strip().strip("/"), path, timeout

    @property
    def label(self):
        return f"the GitHub repository {self.repo}"

    def _request(self, method, body=None):
        url = f"{self.API}/repos/{self.repo}/contents/{quote(self.path)}"
        request = Request(url, method=method, data=None if body is None else json.dumps(body).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "backtester-trade-store",
                                   "Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def load(self):
        try:
            payload = self._request("GET")
        except HTTPError as error:
            if error.code == 404:
                # A missing file is a fresh start; a missing repository or a token without access also returns
                # 404, and the first save then fails with a clear message.
                return empty_data(), None
            raise StoreError(self._explain(error)) from error
        except (OSError, ValueError) as error:
            raise StoreError(f"Could not reach GitHub: {error}") from error
        try:
            text = base64.b64decode(payload.get("content", "")).decode("utf-8")
            return normalize(json.loads(text) if text.strip() else {}), payload.get("sha")
        except ValueError as error:
            raise StoreError(f"{self.path} in {self.repo} is not valid JSON: {error}") from error

    def save(self, data, version=None):
        body = {"message": "Update trading data from the backtester app",
                "content": base64.b64encode(json.dumps(data, indent=1, default=str).encode("utf-8")).decode("ascii")}
        if version:
            body["sha"] = version
        try:
            payload = self._request("PUT", body)
        except HTTPError as error:
            if error.code in (409, 422):  # stale SHA, or the file appeared since it was read
                raise StoreConflict("The saved data changed on another device. Reload it, then make the change "
                                    "again.") from error
            raise StoreError(self._explain(error)) from error
        except (OSError, ValueError) as error:
            raise StoreError(f"Could not reach GitHub: {error}") from error
        return (payload.get("content") or {}).get("sha")

    def _explain(self, error):
        if error.code == 401:
            return "GitHub refused the token (401). It may have expired: make a new one and update the app's secrets."
        if error.code in (403, 404):
            return (f"GitHub said {error.code} for {self.repo}. Check the repository name in the secrets and that "
                    "the token has Contents: Read and write on that repository.")
        return f"GitHub error {error.code}: {error.reason}"


def make_store(config, local_path):
    """A GitHubStore when the secrets name a token and a repository, otherwise a LocalStore."""

    config = config or {}
    token, repo = str(config.get("github_token") or "").strip(), str(config.get("repo") or "").strip()
    if token and repo:
        return GitHubStore(token, repo, str(config.get("path") or "trading.json"))
    return LocalStore(local_path)
