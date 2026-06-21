// omlx/admin/static/js/webmcp/lib/lifecycle.js
//
// AbortController-as-handle helpers and page-unload teardown.
// Phase A registers nothing here. Phase B+ tools (long-running downloads,
// benchmark SSE streams) use this to clean up on navigation.

const handles = new Map(); // id -> AbortController

export function makeHandle(id) {
  const ctrl = new AbortController();
  handles.set(id, ctrl);
  return ctrl;
}

export function dropHandle(id) {
  const ctrl = handles.get(id);
  if (ctrl) {
    ctrl.abort();
    handles.delete(id);
  }
}

window.addEventListener("pagehide", () => {
  for (const ctrl of handles.values()) ctrl.abort();
  handles.clear();
});
