"""Environment interface + a Playwright browser backend.

The Env interface is the seam that matters: the agent loop never imports
Playwright. When the Xvfb desktop backend arrives it implements this same
interface (screenshot -> pixels, act -> effect) and nothing upstream changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .actions import Action


@dataclass
class Observation:
    """What the agent sees after an action."""
    png: bytes
    width: int
    height: int
    # Backend-specific context shown to the model as text (url, title, ...).
    info: dict[str, Any] = field(default_factory=dict)
    # Set when the previous action failed; fed back so the model can recover.
    error: str | None = None


class Env(ABC):
    @abstractmethod
    def reset(self, task: str) -> Observation: ...

    @abstractmethod
    def step(self, action: Action) -> Observation: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BrowserEnv(Env):
    """Chromium via Playwright. Coordinates are viewport pixels."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 800,
        headless: bool = True,
        start_url: str = "about:blank",
        nav_timeout_ms: int = 20_000,
        snap_radius: float = 45.0,
        marks: bool = True,
    ):
        self.width, self.height = width, height
        self.headless = headless
        self.start_url = start_url
        self.nav_timeout_ms = nav_timeout_ms
        # Measured: the model grounds y within a couple of pixels but can be ~40px
        # out in x on dense text, which misses a 24px-wide nav link entirely.
        # The DOM knows where the clickable things are, so snap to the nearest one.
        # Set to 0 to disable and measure raw grounding.
        self.snap_radius = snap_radius
        self.last_snap: str | None = None
        # Set-of-Marks: draw numbered badges on interactive elements so the model
        # can pick element 7 instead of guessing a pixel. Measured on dense nav
        # text, pixel grounding lands on the neighbouring link; an index cannot.
        self.marks = marks
        self._marks: dict[int, dict] = {}
        self._pw = None
        self._browser = None
        self._ctx = None

    # -- lifecycle ---------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._ctx is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        self._ctx = self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=1,  # keep model coords == CSS pixels
        )
        self._ctx.set_default_timeout(self.nav_timeout_ms)

    @property
    def page(self):
        """Newest page, so popups/new tabs become the active target."""
        self._ensure_started()
        pages = [p for p in self._ctx.pages if not p.is_closed()]
        if not pages:
            return self._ctx.new_page()
        return pages[-1]

    def reset(self, task: str = "") -> Observation:
        self.close()
        self._ensure_started()
        page = self._ctx.new_page()
        if self.start_url and self.start_url != "about:blank":
            page.goto(self.start_url, wait_until="domcontentloaded")
        return self.observe()

    def close(self) -> None:
        for obj, meth in ((self._ctx, "close"), (self._browser, "close"), (self._pw, "stop")):
            if obj is not None:
                try:
                    getattr(obj, meth)()
                except Exception:
                    pass
        self._ctx = self._browser = self._pw = None

    # -- observation -------------------------------------------------------
    def observe(self, error: str | None = None) -> Observation:
        page = self.page
        try:
            page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except Exception:
            pass  # a slow/never-idle page is still worth screenshotting
        marks = self._apply_marks() if self.marks else []
        png = page.screenshot(type="png")
        if marks:
            self._clear_marks()  # never let badges leak into the real page
        info: dict[str, Any] = {}
        if marks:
            info["marks"] = marks
        try:
            info.update(
                url=page.url,
                title=page.title(),
                tabs=len([p for p in self._ctx.pages if not p.is_closed()]),
            )
            if self.last_snap:
                info["snap"] = self.last_snap
        except Exception:
            pass
        return Observation(png=png, width=self.width, height=self.height,
                           info=info, error=error)

    # -- set-of-marks ------------------------------------------------------
    _MARK_JS = """() => {
      const SEL = 'a,button,input,select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],[onclick]';
      document.querySelectorAll('.__mark__').forEach(n => n.remove());
      const layer = document.createElement('div');
      layer.className = '__mark__';
      layer.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none';
      document.body.appendChild(layer);
      const COLORS = ['#e11d48','#2563eb','#16a34a','#c2410c','#7c3aed','#0891b2'];
      const out = [];
      let i = 0;
      for (const el of document.querySelectorAll(SEL)) {
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
        const s = getComputedStyle(el);
        if (s.visibility === 'hidden' || s.display === 'none' || +s.opacity === 0) continue;
        // Only mark what a user could actually hit at this point.
        const el2 = document.elementFromPoint(Math.min(Math.max(r.x + r.width/2, 1), innerWidth - 1),
                                              Math.min(Math.max(r.y + r.height/2, 1), innerHeight - 1));
        if (!el2 || (el2 !== el && !el.contains(el2) && !el2.contains(el))) continue;
        const c = COLORS[i % COLORS.length];
        // Keep marks visually quiet. On link-dense pages heavy badges bury the
        // page text, which costs the model more than imprecise aiming does.
        const box = document.createElement('div');
        box.style.cssText = `position:absolute;left:${r.x}px;top:${r.y}px;width:${r.width}px;height:${r.height}px;outline:1px dashed ${c}80;outline-offset:0`;
        const tag = document.createElement('div');
        tag.textContent = i;
        // Sit just outside the top-left corner so the element's own text stays legible.
        tag.style.cssText = `position:absolute;left:${Math.max(r.x-2,0)}px;top:${Math.max(r.y-8,0)}px;background:${c};color:#fff;font:700 9px/9px monospace;padding:0 2px;border-radius:2px;opacity:.9`;
        layer.append(box, tag);
        out.push({id: i, x: r.x, y: r.y, w: r.width, h: r.height,
                  label: (el.innerText || el.value || el.getAttribute('aria-label')
                          || el.getAttribute('placeholder') || el.tagName).trim().slice(0, 50)});
        i++;
      }
      return out;
    }"""

    def _apply_marks(self) -> list[dict]:
        try:
            boxes = self.page.evaluate(self._MARK_JS)
        except Exception:
            return []
        self._marks = {b["id"]: b for b in boxes}
        return boxes

    def _clear_marks(self) -> None:
        try:
            self.page.evaluate("() => document.querySelectorAll('.__mark__').forEach(n => n.remove())")
        except Exception:
            pass

    # -- click snapping ----------------------------------------------------
    _INTERACTIVE_JS = """() => {
      const sel = 'a,button,input,select,textarea,summary,[role=button],[role=link],[role=tab],[role=menuitem],[onclick]';
      return Array.from(document.querySelectorAll(sel)).map(el => {
        const r = el.getBoundingClientRect();
        return {x: r.x, y: r.y, w: r.width, h: r.height,
                label: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 40)};
      }).filter(b => b.w > 1 && b.h > 1 && b.x + b.w > 0 && b.y + b.h > 0
                     && b.x < window.innerWidth && b.y < window.innerHeight);
    }"""

    def _snap(self, x: float, y: float) -> tuple[float, float]:
        """Nudge a predicted point onto the nearest clickable element."""
        self.last_snap = None
        if self.snap_radius <= 0:
            return x, y
        try:
            boxes = self.page.evaluate(self._INTERACTIVE_JS)
        except Exception:
            return x, y

        best, best_d = None, float("inf")
        for b in boxes:
            # Already inside a clickable element -- the model was right, leave it.
            if b["x"] <= x <= b["x"] + b["w"] and b["y"] <= y <= b["y"] + b["h"]:
                return x, y
            # Distance to the box, not its centre: a wide nav bar should not lose
            # to a small button just because its midpoint is far away.
            dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
            dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best, best_d = b, d

        if best is None or best_d > self.snap_radius:
            return x, y
        cx, cy = best["x"] + best["w"] / 2, best["y"] + best["h"] / 2
        self.last_snap = f"snapped {best_d:.0f}px to {best['label'] or 'element'!r}"
        return cx, cy

    # -- actions -----------------------------------------------------------
    def step(self, action: Action) -> Observation:
        try:
            self._apply(action)
        except Exception as e:
            # Never crash the episode on a bad action; report it and let the
            # model see the (unchanged) screen plus the error.
            return self.observe(error=f"{type(e).__name__}: {e}")
        return self.observe()

    def _apply(self, action: Action) -> None:
        page, a = self.page, action.args
        name = action.name

        if name == "click_id":
            mid = int(a["id"])
            if mid not in self._marks:
                raise ValueError(f"no element {mid} on screen "
                                 f"(valid ids: 0-{max(self._marks) if self._marks else -1})")
            b = self._marks[mid]
            page.mouse.click(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2,
                             button=str(a.get("button", "left")))
        elif name == "click":
            x, y = self._snap(float(a["x"]), float(a["y"]))
            page.mouse.click(x, y, button=str(a.get("button", "left")))
        elif name == "double_click":
            x, y = self._snap(float(a["x"]), float(a["y"]))
            page.mouse.dblclick(x, y)
        elif name == "type":
            page.keyboard.type(str(a["text"]), delay=15)
            if a.get("enter"):
                page.keyboard.press("Enter")
        elif name == "key":
            for combo in str(a["keys"]).split():
                page.keyboard.press(combo)
        elif name == "scroll":
            if a.get("x") is not None and a.get("y") is not None:
                page.mouse.move(float(a["x"]), float(a["y"]))
            page.mouse.wheel(float(a.get("dx") or 0), float(a["dy"]))
        elif name == "goto":
            url = str(a["url"])
            if not url.startswith(("http://", "https://", "about:", "file://")):
                url = "https://" + url
            page.goto(url, wait_until="domcontentloaded")
        elif name == "back":
            page.go_back(wait_until="domcontentloaded")
        elif name == "wait":
            page.wait_for_timeout(float(a.get("ms") or 1000))
        else:
            raise ValueError(f"{name} is not executable in BrowserEnv")

        # Let JS-driven UI settle before the next screenshot.
        page.wait_for_timeout(400)
