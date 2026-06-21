// omlx/admin/static/js/webmcp/tools/chat.js
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

export function registerChatTools(mc) {
  mc.registerTool({
    name: "chat.complete",
    description:
      "Send a single user message to the loaded oMLX model and return the " +
      "assistant's response. Requires at least one model to be loaded — call " +
      "models.list first; if empty, bootstrap with models.search and models.download. " +
      "Uses the default model unless 'model' is supplied.",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "string", description: "User message text." },
        model: { type: "string", description: "Optional model ID; omit to use default." },
        temperature: { type: "number", minimum: 0, maximum: 2 },
        max_tokens: { type: "integer", minimum: 1 },
        system: { type: "string", description: "Optional system prompt for this turn." },
      },
      required: ["message"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const messages = [];
        if (args.system) messages.push({ role: "system", content: args.system });
        messages.push({ role: "user", content: args.message });
        const body = { messages, stream: false };
        if (args.model) body.model = args.model;
        if (args.temperature != null) body.temperature = args.temperature;
        if (args.max_tokens != null) body.max_tokens = args.max_tokens;
        const r = await v1Fetch("/v1/chat/completions", {
          method: "POST",
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          if (r.status === 401) return err("UNAUTHORIZED", "API key invalid or missing. Set it in the oMLX chat page.");
          return err("BACKEND_ERROR", r.body?.error?.message || "chat/completions failed", { status: r.status });
        }
        const text = r.body?.choices?.[0]?.message?.content ?? "";
        return ok({ response: text, model: r.body?.model, usage: r.body?.usage });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "chat.complete_with_history",
    description:
      "Like chat.complete but accepts a full message array for multi-turn context. " +
      "Requires a loaded model (same precondition as chat.complete).",
    inputSchema: {
      type: "object",
      properties: {
        messages: {
          type: "array",
          items: {
            type: "object",
            properties: {
              role: { type: "string", enum: ["system", "user", "assistant", "tool"] },
              content: { type: "string" },
            },
            required: ["role", "content"],
            additionalProperties: false,
          },
        },
        model: { type: "string" },
        temperature: { type: "number" },
        max_tokens: { type: "integer" },
      },
      required: ["messages"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const body = { messages: args.messages, stream: false };
        if (args.model) body.model = args.model;
        if (args.temperature != null) body.temperature = args.temperature;
        if (args.max_tokens != null) body.max_tokens = args.max_tokens;
        const r = await v1Fetch("/v1/chat/completions", {
          method: "POST",
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          if (r.status === 401) return err("UNAUTHORIZED", "API key invalid or missing.");
          return err("BACKEND_ERROR", r.body?.error?.message || "chat/completions failed", { status: r.status });
        }
        const text = r.body?.choices?.[0]?.message?.content ?? "";
        return ok({ response: text, model: r.body?.model, usage: r.body?.usage });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "chat.list_models",
    description:
      "List models exposed at /v1/models (the OpenAI-compat endpoint). Only " +
      "currently-loaded/servable models appear here. Use to answer 'what models " +
      "are available right now?'",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await v1Fetch("/v1/models");
        if (!r.ok) return err("BACKEND_ERROR", "v1/models failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  // Page-state tools — only functional on /admin/chat
  mc.registerTool({
    name: "chat.get_active_conversation",
    description:
      "Returns the messages currently visible in the oMLX chat UI. " +
      "Only works when the user is on the /admin/chat page — returns " +
      "PRECONDITION_FAILED on any other page.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      if (!location.pathname.startsWith("/admin/chat")) {
        return err("PRECONDITION_FAILED", "This tool only works on the /admin/chat page.", {
          precondition: "chat_page_active",
          suggested_next: ["chat.complete"],
        });
      }
      try {
        const root = document.querySelector("[x-data]");
        const data = root ? Alpine.$data(root) : null;
        if (!data) return err("INTERNAL", "Could not read Alpine chat state.");
        const messages = (data.messages || data.currentMessages || data.conversation || []).map((m) => ({
          role: m.role,
          content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
        }));
        return ok({ messages, count: messages.length });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "chat.send_to_current",
    description:
      "Append a user message to the current chat conversation and trigger " +
      "a streamed completion — same as clicking Send. Only works on /admin/chat.",
    inputSchema: {
      type: "object",
      properties: { message: { type: "string" } },
      required: ["message"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      if (!location.pathname.startsWith("/admin/chat")) {
        return err("PRECONDITION_FAILED", "This tool only works on the /admin/chat page.", {
          precondition: "chat_page_active",
          suggested_next: ["chat.complete"],
        });
      }
      try {
        const root = document.querySelector("[x-data]");
        const data = root ? Alpine.$data(root) : null;
        if (!data) return err("INTERNAL", "Could not read Alpine chat state.");
        const sendFn = data.sendMessage || data.send || data.submitMessage;
        if (!sendFn) return err("INTERNAL", "Could not find send function in chat state.");
        data.inputMessage = args.message;
        await sendFn.call(data);
        return ok({ sent: true, message: args.message });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "chat.clear_current",
    description:
      "Clear the current conversation in the oMLX chat UI. Only works on /admin/chat.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false },
    async execute(_args) {
      if (!location.pathname.startsWith("/admin/chat")) {
        return err("PRECONDITION_FAILED", "This tool only works on the /admin/chat page.", {
          precondition: "chat_page_active",
        });
      }
      try {
        const root = document.querySelector("[x-data]");
        const data = root ? Alpine.$data(root) : null;
        if (!data) return err("INTERNAL", "Could not read Alpine chat state.");
        const clearFn = data.clearConversation || data.clearChat || data.newConversation;
        if (!clearFn) return err("INTERNAL", "Could not find clear function in chat state.");
        await clearFn.call(data);
        return ok({ cleared: true });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
