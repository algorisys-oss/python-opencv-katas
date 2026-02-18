/**
 * kata-page.tsx — Main kata view with two tabs: Details and Code.
 * Code tab supports: maximize editor, maximize output, open output in new tab.
 * Live camera katas run in "local mode" — launching on the desktop with real camera.
 */

import { Component, createSignal, createResource, createEffect, Show } from "solid-js";
import { useParams } from "@solidjs/router";
import { api } from "../api/client";
import type { ExecuteResult } from "../api/client";
import KataHeader from "../components/kata-header";
import DemoPanel from "../components/demo-panel";
import CodeEditor from "../components/code-editor";
import OutputPanel from "../components/output-panel";

type Tab = "details" | "code";
type PanelFocus = "split" | "editor" | "output";

const KataPage: Component = () => {
    const params = useParams<{ slug: string }>();
    const [kata] = createResource(() => params.slug, api.getKata);

    const [activeTab, setActiveTab] = createSignal<Tab>("details");
    const [code, setCode] = createSignal("");
    const [result, setResult] = createSignal<ExecuteResult | null>(null);
    const [running, setRunning] = createSignal(false);
    const [localRunning, setLocalRunning] = createSignal(false);
    const [focus, setFocus] = createSignal<PanelFocus>("split");

    const starterCode = () => kata()?.starter_code ?? "";
    const isLive = () => kata()?.level === "live";

    // Reset output state when navigating to a different kata
    createEffect(() => {
        params.slug; // track slug changes
        setResult(null);
        setRunning(false);
        setLocalRunning(false);
        setFocus("split");
    });

    // Set code from kata when it loads (handles navigation correctly)
    createEffect(() => {
        const k = kata();
        if (k) {
            setCode(k.starter_code);
        }
    });

    const handleReset = () => {
        setCode(starterCode());
        setResult(null);
    };

    const handleRun = async () => {
        if (running()) return;
        setRunning(true);
        setResult(null);
        try {
            const local = isLive();
            const res = await api.executeCode(code(), local);
            setResult(res);
            if (local) {
                setLocalRunning(true);
            }
        } catch (e: any) {
            setResult({ image_b64: null, logs: "", error: e.message });
        } finally {
            setRunning(false);
        }
    };

    const handleStop = async () => {
        try {
            await api.stopExecution();
            setLocalRunning(false);
            setResult({
                image_b64: null,
                logs: result()?.logs + "\nProcess stopped.",
                error: "",
            });
        } catch (e: any) {
            // ignore
        }
    };

    const openOutputInNewTab = () => {
        const r = result();
        if (!r?.image_b64) return;
        const html = `<!DOCTYPE html>
<html>
<head><title>Output</title><style>
  body { margin: 0; background: #0d1117; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  img { max-width: 100%; max-height: 100vh; image-rendering: pixelated; }
</style></head>
<body><img src="data:image/png;base64,${r.image_b64}" /></body>
</html>`;
        const blob = new Blob([html], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank");
    };

    const toggleFocus = (panel: "editor" | "output") => {
        setFocus(f => f === panel ? "split" : panel);
    };

    return (
        <div class="kata-page">
            <Show when={kata.loading}>
                <div class="kata-loading">Loading kata…</div>
            </Show>

            <Show when={kata.error}>
                <div class="kata-error">Failed to load kata: {String(kata.error)}</div>
            </Show>

            <Show when={kata()}>
                {(k) => (
                    <>
                        <KataHeader kata={k()} />

                        {/* Tab bar */}
                        <div class="kata-tabs">
                            <button
                                class="kata-tab"
                                classList={{ "kata-tab--active": activeTab() === "details" }}
                                onClick={() => setActiveTab("details")}
                            >
                                Details
                            </button>
                            <button
                                class="kata-tab"
                                classList={{ "kata-tab--active": activeTab() === "code" }}
                                onClick={() => setActiveTab("code")}
                            >
                                Code
                            </button>
                        </div>

                        {/* Tab content */}
                        <div class="kata-tab-content">
                            <Show when={activeTab() === "details"}>
                                <DemoPanel kata={k()} />
                            </Show>

                            <Show when={activeTab() === "code"}>
                                <div
                                    class="kata-code-layout"
                                    classList={{
                                        "kata-code-layout--editor-max": focus() === "editor",
                                        "kata-code-layout--output-max": focus() === "output",
                                    }}
                                >
                                    {/* Editor pane */}
                                    <div class="kata-editor-pane">
                                        <div class="kata-editor-toolbar">
                                            <span class="kata-editor-filename">kata.py</span>
                                            <div class="kata-editor-actions">
                                                <button class="btn btn--ghost" onClick={handleReset} title="Reset to starter code">
                                                    Reset
                                                </button>
                                                <button
                                                    class="btn btn--icon"
                                                    classList={{ "btn--icon-active": focus() === "editor" }}
                                                    onClick={() => toggleFocus("editor")}
                                                    title={focus() === "editor" ? "Restore split view" : "Maximize editor"}
                                                >
                                                    {focus() === "editor" ? "⊡" : "⊞"}
                                                </button>
                                                <Show when={localRunning()}>
                                                    <button
                                                        class="btn btn--danger"
                                                        onClick={handleStop}
                                                        title="Stop the running camera process"
                                                    >
                                                        ■ Stop
                                                    </button>
                                                </Show>
                                                <button
                                                    class="btn btn--primary"
                                                    onClick={handleRun}
                                                    disabled={running()}
                                                >
                                                    {running() ? "Launching…" : isLive() ? "▶ Run on Desktop" : "▶ Run"}
                                                </button>
                                            </div>
                                        </div>
                                        <CodeEditor
                                            value={code()}
                                            onChange={setCode}
                                        />
                                    </div>

                                    {/* Output pane */}
                                    <div class="kata-output-pane">
                                        <div class="kata-output-toolbar">
                                            <div class="kata-output-toolbar-left">
                                                <span class="output-panel-title">Output</span>
                                                <Show when={running()}>
                                                    <span class="output-loading-badge">Running…</span>
                                                </Show>
                                                <Show when={localRunning()}>
                                                    <span class="output-loading-badge" style="background: #238636">Running on Desktop</span>
                                                </Show>
                                            </div>
                                            <div class="kata-output-toolbar-actions">
                                                <Show when={result()?.image_b64}>
                                                    <button
                                                        class="btn btn--icon"
                                                        onClick={openOutputInNewTab}
                                                        title="Open output in new tab"
                                                    >
                                                        ↗
                                                    </button>
                                                </Show>
                                                <button
                                                    class="btn btn--icon"
                                                    classList={{ "btn--icon-active": focus() === "output" }}
                                                    onClick={() => toggleFocus("output")}
                                                    title={focus() === "output" ? "Restore split view" : "Maximize output"}
                                                >
                                                    {focus() === "output" ? "⊡" : "⊞"}
                                                </button>
                                            </div>
                                        </div>
                                        <OutputPanel result={result()} loading={running()} />
                                    </div>
                                </div>
                            </Show>
                        </div>
                    </>
                )}
            </Show>
        </div>
    );
};

export default KataPage;
