// omlx/admin/static/js/webmcp/tools/inference.js
import { ok, err } from "../lib/helpers.js";

function getApiKey() {
  return localStorage.getItem("omlx_chat_api_key") || "";
}

async function v1Fetch(path, init) {
  const key = getApiKey();
  const headers = Object.assign({ "Content-Type": "application/json" }, init?.headers || {});
  if (key) headers["Authorization"] = `Bearer ${key}`;
  const resp = await fetch(path, Object.assign({}, init, { headers }));
  const ct = resp.headers.get("content-type") || "";
  const body = ct.includes("application/json") ? await resp.json() : await resp.text();
  return { ok: resp.ok, status: resp.status, body };
}

export function registerInferenceTools(mc) {
  mc.registerTool({
    name: "inference.complete_raw",
    description:
      "POST to /v1/completions (text completion, non-chat). Use when the user " +
      "explicitly wants raw next-token completion of a prompt prefix rather than " +
      "a chat-style response. Requires a loaded model.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string" },
        model: { type: "string" },
        max_tokens: { type: "integer", default: 256 },
        temperature: { type: "number" },
      },
      required: ["prompt"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const body = { prompt: args.prompt, stream: false };
        if (args.model) body.model = args.model;
        if (args.max_tokens != null) body.max_tokens = args.max_tokens;
        if (args.temperature != null) body.temperature = args.temperature;
        const r = await v1Fetch("/v1/completions", { method: "POST", body: JSON.stringify(body) });
        if (!r.ok) {
          if (r.status === 401) return err("UNAUTHORIZED", "API key invalid or missing.");
          return err("BACKEND_ERROR", r.body?.error?.message || "completions failed", { status: r.status });
        }
        const text = r.body?.choices?.[0]?.text ?? "";
        return ok({ completion: text, model: r.body?.model, usage: r.body?.usage });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "inference.embeddings",
    description:
      "Compute embeddings for input text via /v1/embeddings. " +
      "Requires an embedding model to be loaded — check models.list first.",
    inputSchema: {
      type: "object",
      properties: {
        input: {
          anyOf: [
            { type: "string" },
            { type: "array", items: { type: "string" } },
          ],
        },
        model: { type: "string" },
      },
      required: ["input"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const body = { input: args.input };
        if (args.model) body.model = args.model;
        const r = await v1Fetch("/v1/embeddings", { method: "POST", body: JSON.stringify(body) });
        if (!r.ok) {
          if (r.status === 401) return err("UNAUTHORIZED", "API key invalid or missing.");
          return err("BACKEND_ERROR", r.body?.error?.message || "embeddings failed", { status: r.status });
        }
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
