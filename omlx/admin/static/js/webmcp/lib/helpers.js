// omlx/admin/static/js/webmcp/lib/helpers.js

/** MCP-style tool result for plain text or a JSON-serializable value. */
export function ok(text) {
  return {
    content: [{ type: "text", text: typeof text === "string" ? text : JSON.stringify(text) }],
  };
}

/** MCP-style tool result for a structured error. */
export function err(code, message, extra) {
  return {
    content: [{
      type: "text",
      text: JSON.stringify(Object.assign({ error: code, message }, extra || {})),
    }],
    isError: true,
  };
}

/** Same-origin fetch with the admin session cookie. Returns { ok, status, body, err }. */
export async function adminFetch(path, init) {
  const resp = await fetch(path, Object.assign({ credentials: "same-origin" }, init || {}));
  if (resp.status === 401) {
    return {
      ok: false,
      status: 401,
      body: null,
      err: err("UNAUTHORIZED", "Admin session expired; log in again at /admin/"),
    };
  }
  const ct = resp.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await resp.json() : await resp.text();
  return { ok: resp.ok, status: resp.status, body, err: null };
}
