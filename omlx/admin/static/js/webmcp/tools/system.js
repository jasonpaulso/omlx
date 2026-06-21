// omlx/admin/static/js/webmcp/tools/system.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerSystemTools(mc) {
  mc.registerTool({
    name: "system.ping",
    description:
      "Returns 'pong' and the current ISO timestamp. Use to verify the agent " +
      "can reach the oMLX page-side tool surface. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      return ok({ result: "pong", at: new Date().toISOString() });
    },
  });

  mc.registerTool({
    name: "system.info",
    description:
      "Returns hardware and OS info: CPU model, RAM, GPU/Metal, OS version, " +
      "disk free space, oMLX version. Always callable. Use to answer 'what " +
      "hardware do I have?' or check disk space before downloading large models.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const [device, server] = await Promise.all([
          adminFetch("/admin/api/device-info"),
          adminFetch("/admin/api/server-info"),
        ]);
        if (device.err) return device.err;
        if (server.err) return server.err;
        if (!device.ok) return err("ADMIN_API_ERROR", "device-info failed", { status: device.status });
        if (!server.ok) return err("ADMIN_API_ERROR", "server-info failed", { status: server.status });
        return ok({ ...device.body, ...server.body });
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "system.check_for_updates",
    description:
      "Checks GitHub for newer oMLX releases. Returns current version and " +
      "latest available. Read-only — does not install. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/update-check");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "update-check failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "system.restart_server",
    description:
      "Restart the oMLX server. Destructive — interrupts in-flight requests. " +
      "Use only when the user explicitly asks to restart, or after settings " +
      "changes that require it (the response from settings.update_global will say).",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: false },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/server/restart", { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "restart failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
