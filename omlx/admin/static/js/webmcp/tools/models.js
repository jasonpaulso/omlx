// omlx/admin/static/js/webmcp/tools/models.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerModelsTools(mc) {
  mc.registerTool({
    name: "models.list",
    description:
      "List all locally-known models with their state (loaded/unloaded), " +
      "default flag, pin state, size, and ID. Always callable. Returns empty " +
      "array if no models are downloaded — in that case, call models.search to " +
      "find one and models.download to install it.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/models");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "models list failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.search",
    description:
      "Search HuggingFace Hub for MLX-compatible models. Returns a paginated " +
      "list with download counts, likes, size, and capability tags. Always " +
      "callable. Use this to discover models — combine with models.download to install one.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Free-text search terms, e.g. 'Qwen 3 4B coding'." },
        mlx_only: { type: "boolean", default: true, description: "Restrict to MLX-quantized models." },
        sort: { type: "string", enum: ["downloads", "likes", "trending", "recent"], default: "downloads" },
        min_params_b: { type: "number", description: "Minimum model size in billions of parameters." },
        max_params_b: { type: "number" },
        page: { type: "integer", default: 1, minimum: 1 },
      },
      required: ["query"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    async execute(args) {
      try {
        const params = new URLSearchParams({ q: args.query });
        if (args.mlx_only !== false) params.set("mlx_only", "true");
        if (args.sort) params.set("sort", args.sort);
        if (args.min_params_b != null) params.set("min_params_b", String(args.min_params_b));
        if (args.max_params_b != null) params.set("max_params_b", String(args.max_params_b));
        if (args.page) params.set("page", String(args.page));
        const r = await adminFetch(`/admin/api/hf/search?${params}`);
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "search failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.search_recommended",
    description:
      "Returns a curated list of recommended models — trending or popular. " +
      "Always callable. Use when the user says 'pick a good model' without naming one.",
    inputSchema: {
      type: "object",
      properties: {
        tab: { type: "string", enum: ["trending", "popular"], default: "trending" },
      },
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const tab = args.tab || "trending";
        const r = await adminFetch(`/admin/api/hf/recommended?tab=${tab}`);
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "recommended failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.get_info",
    description:
      "Get detailed metadata for a HuggingFace model — model card, files, " +
      "size, license. Use before models.download to confirm size and license.",
    inputSchema: {
      type: "object",
      properties: {
        repo_id: { type: "string", description: "HF repo ID like 'mlx-community/Qwen3-4B-MLX-4bit'." },
      },
      required: ["repo_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/hf/model-info?repo_id=${encodeURIComponent(args.repo_id)}`);
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "model-info failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.download",
    description:
      "Start downloading a model from HuggingFace. Returns a task_id. " +
      "Poll models.download_status until status === 'completed' or 'failed'. " +
      "Long-running; a 4-bit 4B model is ≈ 2.4 GB. Always callable.",
    inputSchema: {
      type: "object",
      properties: {
        repo_id: { type: "string" },
        token: { type: "string", description: "Optional HF token for gated models." },
      },
      required: ["repo_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const body = { repo_id: args.repo_id };
        if (args.token) body.token = args.token;
        const r = await adminFetch("/admin/api/hf/download", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "download failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.download_status",
    description:
      "List active and recent download tasks with their progress. Always callable. " +
      "Use to poll until downloads complete (status === 'completed' or 'failed').",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/hf/tasks");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "tasks failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.download_cancel",
    description: "Cancel a running download task by task_id.",
    inputSchema: {
      type: "object",
      properties: { task_id: { type: "string" } },
      required: ["task_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/hf/cancel/${encodeURIComponent(args.task_id)}`, { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "cancel failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.load",
    description:
      "Load a downloaded model into memory. Requires the model to be in " +
      "models.list and not already loaded. Synchronous; can take 10-60s for " +
      "large models. If no model is downloaded, use models.search then models.download first.",
    inputSchema: {
      type: "object",
      properties: { model_id: { type: "string" } },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/models/${encodeURIComponent(args.model_id)}/load`, { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "load failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.unload",
    description:
      "Unload a model from memory to free RAM. Use to switch between models " +
      "or before loading a larger one.",
    inputSchema: {
      type: "object",
      properties: { model_id: { type: "string" } },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/models/${encodeURIComponent(args.model_id)}/unload`, { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "unload failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.reload",
    description:
      "Rescan model directories and re-read per-model settings. Use after " +
      "adding a model on disk outside oMLX, or after editing model_dirs in settings.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/reload", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "reload failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.set_pinned",
    description:
      "Pin or unpin a model. Pinned models auto-load on server start and survive memory pressure.",
    inputSchema: {
      type: "object",
      properties: {
        model_id: { type: "string" },
        pinned: { type: "boolean" },
      },
      required: ["model_id", "pinned"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/models/${encodeURIComponent(args.model_id)}/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_pinned: args.pinned }),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "set_pinned failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.set_default",
    description:
      "Mark a model as the default. Requests to /v1/* without an explicit model use this.",
    inputSchema: {
      type: "object",
      properties: { model_id: { type: "string" } },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/models/${encodeURIComponent(args.model_id)}/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_default: true }),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "set_default failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.update_settings",
    description:
      "Update per-model sampling settings and metadata. Pass only the fields " +
      "you want to change; others are preserved. Call models.list_profile_fields " +
      "to discover all available keys.",
    inputSchema: {
      type: "object",
      properties: {
        model_id: { type: "string" },
        temperature: { type: "number" },
        top_p: { type: "number" },
        top_k: { type: "integer" },
        max_tokens: { type: "integer" },
        max_context_window: { type: "integer" },
        repetition_penalty: { type: "number" },
        system_prompt: { type: "string" },
      },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const { model_id, ...settings } = args;
        const r = await adminFetch(`/admin/api/models/${encodeURIComponent(model_id)}/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "update_settings failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.delete_local",
    description:
      "Delete a downloaded model from local cache. Destructive and irreversible — " +
      "confirm with the user before calling. model_name is the HF repo ID.",
    inputSchema: {
      type: "object",
      properties: { model_name: { type: "string" } },
      required: ["model_name"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false, destructiveHint: true },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/hf/models/${encodeURIComponent(args.model_name)}`, { method: "DELETE" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "delete failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.list_local_hf",
    description:
      "List locally-cached HuggingFace models with their on-disk sizes. " +
      "Use to find disk hogs before deletion. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/hf/models");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "list_local_hf failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "models.list_profile_fields",
    description:
      "Returns metadata about all per-model profile fields: name, type, " +
      "default, valid range. Use to discover what models.update_settings accepts. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/profile-fields");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "profile-fields failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
