// omlx/admin/static/js/webmcp/tools/benchmark.js
import { ok, err, adminFetch } from "../lib/helpers.js";

export function registerBenchmarkTools(mc) {
  mc.registerTool({
    name: "benchmark.start_throughput",
    description:
      "Start a throughput benchmark on a loaded model. Returns a bench_id. " +
      "Requires the model to be loaded — call models.load first. Long-running; " +
      "poll benchmark.get_status until completed. Typical duration: short ≈ 10s, " +
      "medium ≈ 60s, long ≈ 300s.",
    inputSchema: {
      type: "object",
      properties: {
        model_id: { type: "string" },
        preset: { type: "string", enum: ["short", "medium", "long"], description: "Benchmark duration preset." },
        prompt_tokens: { type: "integer", description: "Custom prompt size (overrides preset)." },
        max_tokens: { type: "integer", description: "Custom generation length (overrides preset)." },
      },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const body = { model_id: args.model_id };
        if (args.preset) body.preset = args.preset;
        if (args.prompt_tokens != null) body.prompt_tokens = args.prompt_tokens;
        if (args.max_tokens != null) body.max_tokens = args.max_tokens;
        const r = await adminFetch("/admin/api/bench/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "bench start failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.get_status",
    description:
      "List currently active throughput benchmarks with progress. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/bench/active");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "bench active failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.get_results",
    description:
      "Get the results of a completed throughput benchmark — prefill tok/s, " +
      "decode tok/s, latencies, memory peak.",
    inputSchema: {
      type: "object",
      properties: { bench_id: { type: "string" } },
      required: ["bench_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: true },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/bench/${encodeURIComponent(args.bench_id)}/results`);
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "bench results failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.cancel",
    description: "Cancel a running throughput benchmark by bench_id.",
    inputSchema: {
      type: "object",
      properties: { bench_id: { type: "string" } },
      required: ["bench_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const r = await adminFetch(`/admin/api/bench/${encodeURIComponent(args.bench_id)}/cancel`, { method: "POST" });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "bench cancel failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.queue_accuracy",
    description:
      "Queue an accuracy benchmark (perplexity / MMLU-style eval). Returns " +
      "immediately with a queue position. Poll benchmark.accuracy_status to track progress.",
    inputSchema: {
      type: "object",
      properties: {
        model_id: { type: "string" },
        dataset: { type: "string", description: "Eval dataset name." },
      },
      required: ["model_id"],
      additionalProperties: false,
    },
    annotations: { readOnlyHint: false },
    async execute(args) {
      try {
        const body = { model_id: args.model_id };
        if (args.dataset) body.dataset = args.dataset;
        const r = await adminFetch("/admin/api/bench/accuracy/queue/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", r.body?.detail || "accuracy queue failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.accuracy_status",
    description:
      "Get the accuracy benchmark queue and currently running task. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/bench/accuracy/queue/status");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "accuracy status failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });

  mc.registerTool({
    name: "benchmark.accuracy_results",
    description:
      "Get the latest accuracy benchmark results across all models. Always callable.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
    async execute(_args) {
      try {
        const r = await adminFetch("/admin/api/bench/accuracy/results");
        if (r.err) return r.err;
        if (!r.ok) return err("ADMIN_API_ERROR", "accuracy results failed", { status: r.status });
        return ok(r.body);
      } catch (e) {
        return err("INTERNAL", e.message);
      }
    },
  });
}
