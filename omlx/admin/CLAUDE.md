# Admin UI conventions

Server-rendered Jinja + Alpine.js, **not** a SPA. No build step, no bundler, no npm. `base.html` is extended by every page; `dashboard.js` (~237 KB) is the Alpine root. Every UI action is `@click`→JS method→`fetch('/admin/api/...', {credentials:'same-origin'})`.

CDN dependencies are **vendored offline** into `omlx/admin/static/` via `omlx/admin/vendor_deps.py` — the `JS_DEPS`/`CSS_DEPS` dicts are the source of truth. To add or bump a vendored dep, edit those dicts and run `python omlx/admin/vendor_deps.py`; never hand-edit a vendored file or leave a live CDN URL in a template. The static handler is a route (not a `StaticFiles` mount), so anything under `static/` is served at `/admin/static/...` with no config change.

The WebMCP layer (`static/js/webmcp/`) is a parallel ES-module tree loaded from `base.html`; it does not touch `dashboard.js` or the inline partial scripts. Note `.gitignore` has a broad Python `lib/` rule — `webmcp/lib/` is kept tracked by an explicit negation; new ignored-by-default paths under it need the same treatment.
