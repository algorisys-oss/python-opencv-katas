/**
 * code-editor.tsx — Monaco Editor wrapper for SolidJS.
 * Loads Monaco via CDN (no bundling needed).
 */

import { Component, onMount, onCleanup, createEffect } from "solid-js";

interface Props {
    value: string;
    onChange: (code: string) => void;
    readOnly?: boolean;
}

declare global {
    interface Window {
        monaco: any;
        require: any;
    }
}

const MONACO_CDN = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs";

function loadMonaco(): Promise<void> {
    return new Promise((resolve) => {
        if (window.monaco) return resolve();

        const script = document.createElement("script");
        script.src = `${MONACO_CDN}/loader.js`;
        script.onload = () => {
            window.require.config({ paths: { vs: MONACO_CDN } });
            window.require(["vs/editor/editor.main"], () => resolve());
        };
        document.head.appendChild(script);
    });
}

const CodeEditor: Component<Props> = (props) => {
    let containerRef: HTMLDivElement | undefined;
    let editor: any;

    onMount(async () => {
        await loadMonaco();

        editor = window.monaco.editor.create(containerRef!, {
            value: props.value,
            language: "python",
            theme: "vs-dark",
            fontSize: 14,
            fontFamily: "'JetBrains Mono', monospace",
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            lineNumbers: "on",
            renderLineHighlight: "line",
            padding: { top: 12, bottom: 12 },
            readOnly: props.readOnly ?? false,
            automaticLayout: true,
            tabSize: 4,
            insertSpaces: true,
        });

        editor.onDidChangeModelContent(() => {
            props.onChange(editor.getValue());
        });
    });

    // Sync external value changes (e.g. kata switch / reset)
    createEffect(() => {
        if (editor && editor.getValue() !== props.value) {
            editor.setValue(props.value);
        }
    });

    onCleanup(() => editor?.dispose());

    return <div ref={containerRef} class="code-editor-container" />;
};

export default CodeEditor;
