// omlx/admin/static/js/webmcp/tools/settings.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerSettingsTools(mc) {
  mc.registerTool({
    name: "settings.get_global",
    description:
      "Returns the entire global settings tree: server config, model dirs, " +
      "memory guard, scheduler, cache, sampling defaults, auth, integrations, etc. " +
      "Always callable. Use to inspect current config before calling settings.update_global.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/global-settings");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "get global-settings failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "settings.update_global",
    description:
      "Update one or more global settings. Pass a partial settings object — " +
      "only changed fields are persisted. Some changes require system.restart_server " +
      "to take effect (the response will indicate this). Call settings.get_global first " +
      "to see current values.",
    inputSchema: {
      type: "object",
      properties: {
        server: {
          type: "object",
          properties: {
            host: { type: "string" },
            port: { type: "integer", minimum: 1, maximum: 65535 },
            log_level: { type: "string", enum: ["debug", "info", "warning", "error"] },
          },
          additionalProperties: false,
        },
        model: {
          type: "object",
          properties: {
            model_dirs: { type: "array", items: { type: "string" } },
          },
          additionalProperties: false,
        },
        memory: {
          type: "object",
          properties: {
            prefill_memory_guard: { type: "boolean" },
            memory_guard_tier: { type: "string", enum: ["aggressive", "balanced", "permissive"] },
            memory_guard_custom_ceiling_gb: { type: "number" },
          },
          additionalProperties: false,
        },
        scheduler: {
          type: "object",
          properties: {
            max_concurrent_requests: { type: "integer" },
            chunked_prefill: { type: "boolean" },
          },
          additionalProperties: false,
        },
        sampling: {
          type: "object",
          properties: {
            temperature: { type: "number" },
            top_p: { type: "number" },
            top_k: { type: "integer" },
            max_tokens: { type: "integer" },
            max_context_window: { type: "integer" },
          },
          additionalProperties: false,
        },
        huggingface: {
          type: "object",
          properties: {
            endpoint: { type: "string" },
            hf_cache_enabled: { type: "boolean" },
            hf_cache_path: { type: "string" },
          },
          additionalProperties: false,
        },
      },
      additionalProperties: true,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch("/admin/api/global-settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(args),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "update global-settings failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "settings.create_sub_key",
    description:
      "Create a subordinate API key (for API use only — cannot log in to admin). " +
      "Returns the generated key once; it cannot be retrieved later. Store it securely.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Label for the key, e.g. 'Claude Code'." },
      },
      required: ["name"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch("/admin/api/sub-keys", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: args.name }),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "create sub-key failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "settings.delete_sub_key",
    description: "Delete a subordinate API key by its key value.",
    inputSchema: {
      type: "object",
      properties: { key: { type: "string" } },
      required: ["key"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, destructiveHint: true },
    async execute(args) {
      try {
        const r = await adminFetch("/admin/api/sub-keys", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: args.key }),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "delete sub-key failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
