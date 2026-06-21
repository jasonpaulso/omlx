// omlx/admin/static/js/webmcp/tools/logs.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerLogsTools(mc) {
  mc.registerTool({
    name: "logs.tail",
    description:
      "Return the most recent N log lines. Always callable. Use to diagnose " +
      "'why didn't X work?' — the answer is usually in here. Filter by min_level " +
      "to suppress noise. Logs may contain user-supplied prompt text.",
    inputSchema: {
      type: "object",
      properties: {
        lines: { type: "integer", minimum: 1, maximum: 5000, default: 200 },
        min_level: { type: "string", enum: ["debug", "info", "warning", "error"], default: "info" },
      },
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    async execute(args) {
      try {
        const params = new URLSearchParams();
        if (args.lines) params.set("lines", String(args.lines));
        if (args.min_level) params.set("min_level", args.min_level);
        const r = await adminFetch(`/admin/api/logs?${params}`);
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "logs failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "logs.get_stats",
    description:
      "Get server metrics: total requests, tokens, latencies, current memory, " +
      "queue depth. Use to answer 'how is the server doing?' Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/stats");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "stats failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
