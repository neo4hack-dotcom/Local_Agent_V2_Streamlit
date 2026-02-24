from __future__ import annotations

import json
import re
from typing import Any, Literal
from urllib.parse import urlparse

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm_client import LLMClient

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+", flags=re.IGNORECASE)
_DOMAIN_PATTERN = re.compile(
    r"\b((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/[^\s<>\"]*)?)",
    flags=re.IGNORECASE,
)
_NAVIGABLE_ACTIONS = {"navigate", "click", "fill", "extract", "finish"}
_NAVIGATOR_RUNTIME_CACHE: tuple[bool, str | None] | None = None


def web_navigator_runtime_status(*, force_refresh: bool = False) -> tuple[bool, str | None]:
    global _NAVIGATOR_RUNTIME_CACHE
    if _NAVIGATOR_RUNTIME_CACHE is not None and not force_refresh:
        return _NAVIGATOR_RUNTIME_CACHE

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _NAVIGATOR_RUNTIME_CACHE = (
            False,
            "Playwright Python package is not installed in backend environment.",
        )
        return _NAVIGATOR_RUNTIME_CACHE

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message:
            _NAVIGATOR_RUNTIME_CACHE = (
                False,
                "Playwright browser binaries are missing. "
                "Run 'python -m playwright install chromium' in backend venv.",
            )
            return _NAVIGATOR_RUNTIME_CACHE
        _NAVIGATOR_RUNTIME_CACHE = (
            False,
            f"Playwright runtime unavailable: {message}",
        )
        return _NAVIGATOR_RUNTIME_CACHE

    _NAVIGATOR_RUNTIME_CACHE = (True, None)
    return _NAVIGATOR_RUNTIME_CACHE


class NavigationAction(TypedDict, total=False):
    action: Literal["navigate", "click", "fill", "extract", "finish"]
    url: str
    selector: str
    text: str
    reason: str
    result: str


class NavigationState(TypedDict, total=False):
    task: str
    start_url: str
    url: str
    page_content: str
    history: list[str]
    status: Literal["en_cours", "success", "error", "exhausted"]
    error: str
    step: int
    max_steps: int
    allowed_domains: list[str]
    next_action: NavigationAction
    final_answer: str
    final_rows: list[dict[str, Any]]
    browser: "_BrowserSession"


class _BrowserSession:
    def __init__(self, *, headless: bool, timeout_ms: int, capture_html_chars: int) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.capture_html_chars = capture_html_chars
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ValueError(
                "Web Navigator requires Playwright in backend environment. "
                "Run setup, then 'python -m playwright install chromium' in backend venv."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(ignore_https_errors=True)
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def navigate_to(self, url: str) -> str:
        if not self._page:
            raise RuntimeError("Browser page is not initialized.")
        self._page.goto(url, wait_until="domcontentloaded")
        return self.current_url()

    def fill_form(self, selector: str, text: str) -> None:
        if not self._page:
            raise RuntimeError("Browser page is not initialized.")
        self._page.locator(selector).first.fill(text)

    def click_element(self, selector: str) -> None:
        if not self._page:
            raise RuntimeError("Browser page is not initialized.")
        self._page.locator(selector).first.click()

    def current_url(self) -> str:
        if not self._page:
            return ""
        return str(self._page.url or "").strip()

    def simplified_dom(self) -> str:
        if not self._page:
            return ""
        title = (self._page.title() or "").strip()
        body_text = self._page.locator("body").inner_text(timeout=self.timeout_ms)
        compact = re.sub(r"\s+", " ", body_text).strip()
        base = f"Title: {title}\nURL: {self.current_url()}\nContent:\n{compact}"
        return base[: self.capture_html_chars]


class WebNavigationRunner:
    def __init__(
        self,
        *,
        llm: LLMClient,
        system_prompt: str,
        max_content_chars: int = 7000,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_content_chars = max(500, min(30000, max_content_chars))

        graph = StateGraph(NavigationState)
        graph.add_node("bootstrap", self._bootstrap_node)
        graph.add_node("read_page", self._read_page_node)
        graph.add_node("plan_action", self._plan_action_node)
        graph.add_node("execute_action", self._execute_action_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "bootstrap")
        graph.add_edge("bootstrap", "read_page")
        graph.add_edge("read_page", "plan_action")
        graph.add_conditional_edges(
            "plan_action",
            self._route_after_plan,
            {"execute": "execute_action", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "execute_action",
            self._route_after_execute,
            {"read_page": "read_page", "finalize": "finalize"},
        )
        graph.add_edge("finalize", END)
        self.compiled = graph.compile()

    def run(self, *, task: str, config: dict[str, Any]) -> dict[str, Any]:
        configured_start_url = str(config.get("start_url", "")).strip()
        inferred_start_url = self._extract_first_url(task)

        start_url = configured_start_url
        if self._is_placeholder_start_url(configured_start_url) and inferred_start_url:
            start_url = inferred_start_url
        if not start_url:
            start_url = inferred_start_url
        if not start_url:
            raise ValueError(
                "No start URL found. Set template_config.start_url or include a URL in the request."
            )

        headless = self._to_bool(config.get("headless"), default=True)
        timeout_ms = self._to_int(config.get("timeout_ms"), default=15000, minimum=2000, maximum=120000)
        max_steps = self._to_int(config.get("max_steps"), default=8, minimum=1, maximum=30)
        capture_html_chars = self._to_int(
            config.get("capture_html_chars"),
            default=self.max_content_chars,
            minimum=500,
            maximum=30000,
        )
        allowed_domains = self._normalize_domains(config.get("allowed_domains", []))

        browser = _BrowserSession(
            headless=headless,
            timeout_ms=timeout_ms,
            capture_html_chars=capture_html_chars,
        )
        try:
            browser.start()
            initial_state: NavigationState = {
                "task": task,
                "start_url": start_url,
                "url": "",
                "page_content": "",
                "history": [],
                "status": "en_cours",
                "error": "",
                "step": 0,
                "max_steps": max_steps,
                "allowed_domains": allowed_domains,
                "browser": browser,
            }
            final_state = self.compiled.invoke(initial_state)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Executable doesn't exist" in message or "playwright install" in message:
                raise ValueError(
                    "Web Navigator requires Playwright browser binaries. "
                    "Run 'python -m playwright install chromium' in backend venv."
                ) from exc
            raise
        finally:
            browser.close()

        rows = final_state.get("final_rows", [])
        answer = final_state.get("final_answer", "")
        details = {
            "status": final_state.get("status", "error"),
            "current_url": final_state.get("url", ""),
            "steps": final_state.get("step", 0),
            "history": final_state.get("history", []),
            "error": final_state.get("error", "") or None,
            "start_url": start_url,
            "configured_allowed_domains": allowed_domains,
            "allowed_domains_enforced": False,
        }
        return {"answer": answer, "rows": rows, "details": details}

    def _bootstrap_node(self, state: NavigationState) -> NavigationState:
        start_url = self._normalize_http_url(state.get("start_url", ""))
        if not start_url:
            return {"status": "error", "error": "Invalid start URL."}

        browser = state["browser"]
        try:
            current = browser.navigate_to(start_url)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Navigation failed: {exc}"}

        return {"url": current, "history": [f"navigate:{current}"], "status": "en_cours"}

    def _read_page_node(self, state: NavigationState) -> NavigationState:
        if state.get("status") != "en_cours":
            return {}
        browser = state["browser"]
        try:
            return {"url": browser.current_url(), "page_content": browser.simplified_dom()}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Failed to read page content: {exc}"}

    def _plan_action_node(self, state: NavigationState) -> NavigationState:
        if state.get("status") != "en_cours":
            return {"next_action": {"action": "finish"}}
        if int(state.get("step", 0)) >= int(state.get("max_steps", 0)):
            return {
                "status": "exhausted",
                "error": "Maximum navigation steps reached.",
                "next_action": {"action": "finish"},
            }

        task = state.get("task", "")
        url = state.get("url", "")
        page_content = state.get("page_content", "")
        history = state.get("history", [])

        prompt = (
            "You are planning the next browser action.\n"
            "Return JSON only with this schema:\n"
            '{ "action":"navigate|click|fill|extract|finish", "url":"", "selector":"", "text":"", "reason":"", "result":"" }\n'
            f"Task: {task}\n"
            f"Current URL: {url}\n"
            f"History: {json.dumps(history, ensure_ascii=False)}\n"
            f"Simplified page content:\n{page_content}\n"
            "Rules:\n"
            "- Use navigate with url for page changes.\n"
            "- Cross-domain navigation is allowed when useful for task completion.\n"
            "- Use fill with selector and text for inputs.\n"
            "- Use click with selector for buttons/links.\n"
            "- Use extract when enough context is visible.\n"
            "- Use finish when task is complete or blocked; include result summary.\n"
        )
        raw = self.llm.generate(prompt, system_prompt=self.system_prompt)
        action = self._parse_action(raw)
        return {"next_action": action}

    def _execute_action_node(self, state: NavigationState) -> NavigationState:
        if state.get("status") != "en_cours":
            return {}

        action = state.get("next_action", {})
        action_name = str(action.get("action", "")).strip().lower()
        history = list(state.get("history", []))
        browser = state["browser"]
        step = int(state.get("step", 0)) + 1

        try:
            if action_name == "navigate":
                target_url = self._normalize_http_url(str(action.get("url", "")).strip())
                if not target_url:
                    raise ValueError("Navigate action requires a valid URL.")
                browser.navigate_to(target_url)
                history.append(f"step#{step}:navigate:{target_url}")
                return {"step": step, "history": history, "url": browser.current_url()}

            if action_name == "click":
                selector = str(action.get("selector", "")).strip()
                if not selector:
                    raise ValueError("Click action requires a CSS selector.")
                browser.click_element(selector)
                history.append(f"step#{step}:click:{selector}")
                return {"step": step, "history": history, "url": browser.current_url()}

            if action_name == "fill":
                selector = str(action.get("selector", "")).strip()
                text = str(action.get("text", ""))
                if not selector:
                    raise ValueError("Fill action requires a CSS selector.")
                browser.fill_form(selector, text)
                history.append(f"step#{step}:fill:{selector}:{text[:120]}")
                return {"step": step, "history": history, "url": browser.current_url()}

            if action_name == "extract":
                history.append(f"step#{step}:extract")
                return {"step": step, "history": history}

            if action_name == "finish":
                reason = str(action.get("result", "")).strip() or str(action.get("reason", "")).strip()
                history.append(f"step#{step}:finish:{reason[:200]}")
                return {"step": step, "history": history, "status": "success"}

            raise ValueError(f"Unsupported browser action: {action_name}")
        except Exception as exc:  # noqa: BLE001
            history.append(f"step#{step}:error:{exc}")
            return {"step": step, "history": history, "status": "error", "error": str(exc)}

    @staticmethod
    def _route_after_plan(state: NavigationState) -> str:
        if state.get("status") in {"error", "exhausted", "success"}:
            return "finalize"
        action = str(state.get("next_action", {}).get("action", "")).strip().lower()
        if action == "finish":
            return "finalize"
        return "execute"

    @staticmethod
    def _route_after_execute(state: NavigationState) -> str:
        if state.get("status") in {"error", "exhausted", "success"}:
            return "finalize"
        return "read_page"

    def _finalize_node(self, state: NavigationState) -> NavigationState:
        status = state.get("status", "error")
        history = state.get("history", [])
        page_content = state.get("page_content", "")
        url = state.get("url", "")
        task = state.get("task", "")
        error = state.get("error", "")

        summary_prompt = (
            f"Task: {task}\n"
            f"Status: {status}\n"
            f"Current URL: {url}\n"
            f"Error (if any): {error}\n"
            f"Actions history: {json.dumps(history, ensure_ascii=False)}\n"
            f"Final page content snapshot:\n{page_content}\n"
            "Write a concise final report in English:\n"
            "- What was done\n"
            "- Current page status\n"
            "- Remaining blockers (if any)\n"
        )
        answer = self.llm.generate(summary_prompt, system_prompt=self.system_prompt)

        rows: list[dict[str, Any]] = [
            {"kind": "history", "index": index + 1, "value": entry}
            for index, entry in enumerate(history)
        ]
        rows.append({"kind": "final_page", "url": url, "content_preview": page_content[:3000]})
        return {"final_answer": answer, "final_rows": rows}

    @staticmethod
    def _parse_action(raw_text: str) -> NavigationAction:
        parsed = WebNavigationRunner._parse_json(raw_text)
        if isinstance(parsed, dict):
            action = str(parsed.get("action", "")).strip().lower()
            aliases = {
                "go": "navigate",
                "open": "navigate",
                "goto": "navigate",
                "type": "fill",
                "input": "fill",
                "submit": "click",
                "done": "finish",
                "stop": "finish",
            }
            action = aliases.get(action, action)
            if action not in _NAVIGABLE_ACTIONS:
                action = "extract"
            return {
                "action": action,
                "url": str(parsed.get("url", "")).strip(),
                "selector": str(parsed.get("selector", "")).strip(),
                "text": str(parsed.get("text", "")),
                "reason": str(parsed.get("reason", "")).strip(),
                "result": str(parsed.get("result", "")).strip(),
            }
        return {"action": "extract", "reason": "Fallback action due to invalid planner output."}

    @staticmethod
    def _parse_json(raw_text: str) -> Any:
        cleaned = raw_text.strip()
        match = _JSON_FENCE.search(cleaned)
        if match:
            cleaned = match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _extract_first_url(text: str) -> str:
        for match in _URL_PATTERN.findall(text):
            normalized = WebNavigationRunner._normalize_http_url(match)
            if normalized:
                return normalized
        for match in _DOMAIN_PATTERN.findall(text):
            normalized = WebNavigationRunner._normalize_http_url(match)
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _normalize_http_url(raw: str) -> str:
        value = raw.strip().strip("\"'()[]{}<>,.;")
        if not value:
            return ""
        if re.fullmatch(_DOMAIN_PATTERN, value):
            value = f"https://{value}"
        elif value.lower().startswith("www."):
            value = f"https://{value}"

        parsed = urlparse(value)
        if not parsed.scheme and parsed.path and not parsed.netloc:
            value = f"https://{value}"
            parsed = urlparse(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""
        return parsed._replace(fragment="").geturl()

    @staticmethod
    def _is_placeholder_start_url(raw: str) -> bool:
        normalized = WebNavigationRunner._normalize_http_url(raw)
        return normalized in {
            "https://example.com",
            "https://www.example.com",
            "http://example.com",
            "http://www.example.com",
        }

    @staticmethod
    def _normalize_domains(raw: Any) -> list[str]:
        if isinstance(raw, str):
            values = [item.strip().lower() for item in raw.split(",")]
        elif isinstance(raw, list):
            values = [str(item).strip().lower() for item in raw]
        else:
            values = []
        return [value.lstrip(".") for value in values if value]

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            return default
        if value is None:
            return default
        return bool(value)

    @staticmethod
    def _to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))
