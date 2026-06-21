// omlx/admin/static/js/webmcp/tools/mcp_config.js
import { ok, err } from "../lib/helpers.js";

function getApiKey() {
  return localStorage.getItem("omlx_chat_api_key") || "";
}

async function v1Fetch(path) {
  const key = getApiKey();
  const headers = {};
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const resp = await fetch(path, { headers });
  const ct = resp.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await resp.json() : await resp.text();
  return { ok: resp.ok, status: resp.status, body };
}

export function registerMcpConfigTools(mc) {
  mc.registerTool({
    name: "mcp_config.list_servers",
    description:
      "List MCP servers currently configured for oMLX's chat-model tool use " +
      "(the inbound-tools direction). Returns connection state and tool count. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await v1Fetch("/v1/mcp/servers");
        if (!r.ok) return err("BACKEND_ERROR", "mcp/servers failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "mcp_config.list_tools",
    description:
      "List MCP tools currently exposed to oMLX's chat models. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await v1Fetch("/v1/mcp/tools");
        if (!r.ok) return err("BACKEND_ERROR", "mcp/tools failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
