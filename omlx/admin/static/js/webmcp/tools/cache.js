// omlx/admin/static/js/webmcp/tools/cache.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerCacheTools(mc) {
  mc.registerTool({
    name: "cache.clear_ssd",
    description:
      "Clear the on-disk KV cache (SSD). Destructive — frees disk but invalidates " +
      "prefix caching. Confirm with user before calling.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false, destructiveHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/ssd-cache/clear", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "ssd-cache clear failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "cache.clear_hot",
    description:
      "Clear the in-memory hot cache. Destructive — frees RAM but invalidates " +
      "recent prefix caching. Confirm with user before calling.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false, destructiveHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/hot-cache/clear", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "hot-cache clear failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "cache.clear_stats",
    description:
      "Reset server metrics for the current session. Does not touch alltime metrics.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/stats/clear", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "stats clear failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "cache.clear_alltime_stats",
    description:
      "Reset alltime server metrics. Irreversible. Confirm with user before calling.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false, destructiveHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/stats/clear-alltime", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "stats clear-alltime failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
