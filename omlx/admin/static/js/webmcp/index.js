// omlx/admin/static/js/webmcp/index.js
//
// Registers all WebMCP tools on every admin page. The @mcp-b/global polyfill
// is loaded synchronously in base.html <head>, so navigator.modelContext is
// already available when this module runs.

import { registerSystemTools }    from "./tools/system.js";
import { registerModelsTools }    from "./tools/models.js";
import { registerChatTools }      from "./tools/chat.js";
import { registerInferenceTools } from "./tools/inference.js";
import { registerSettingsTools }  from "./tools/settings.js";
import { registerBenchmarkTools } from "./tools/benchmark.js";
import { registerLogsTools }      from "./tools/logs.js";
import { registerCacheTools }     from "./tools/cache.js";
import { registerMcpConfigTools } from "./tools/mcp_config.js";

async function main() {
  const mc = navigator.modelContext;
  if (!mc) {
    console.warn("[webmcp] navigator.modelContext unavailable — polyfill may not have loaded");
    return;
  }

  registerSystemTools(mc);
  registerModelsTools(mc);
  registerChatTools(mc);
  registerInferenceTools(mc);
  registerSettingsTools(mc);
  registerBenchmarkTools(mc);
  registerLogsTools(mc);
  registerCacheTools(mc);
  registerMcpConfigTools(mc);

  // Expose tool manifest two ways for browser agents that discover via
  // window inspection or DOM reading rather than calling listTools() directly.
  const tools = await mc.listTools();
  const manifest = tools.map(t => ({ name: t.name, description: t.description }));
  window.__omlx_tools__ = manifest;

  const el = document.createElement("script");
  el.type = "application/json";
  el.id = "omlx-webmcp-manifest";
  el.textContent = JSON.stringify({
    webmcp: true,
    description:
      "This oMLX admin page exposes " + manifest.length + " WebMCP tools via " +
      "navigator.modelContext. Call navigator.modelContext.listTools() to enumerate " +
      "them, or read this element for a summary. Tools cover: model management, " +
      "chat/inference, settings, benchmarks, logs, cache, and MCP config.",
    tools: manifest,
  }, null, 2);
  document.head.appendChild(el);
}

main();
