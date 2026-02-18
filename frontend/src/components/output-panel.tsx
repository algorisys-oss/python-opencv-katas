/**
 * output-panel.tsx — Displays execution result: image, logs, and errors.
 * The panel header/toolbar is owned by the parent (kata-page.tsx).
 */

import { Component, Show } from "solid-js";
import type { ExecuteResult } from "../api/client";

interface Props {
    result: ExecuteResult | null;
    loading: boolean;
}

const OutputPanel: Component<Props> = (props) => {
    return (
        <div class="output-panel-body">
            <Show
                when={!props.loading && props.result}
                fallback={
                    <div class="output-empty">
                        <span class="output-empty-icon">▶</span>
                        <p class="output-empty-text">Run your code to see the output here.</p>
                    </div>
                }
            >
                {/* Image output */}
                <Show when={props.result?.image_b64}>
                    <div class="output-image-wrapper">
                        <img
                            src={`data:image/png;base64,${props.result!.image_b64}`}
                            alt="Code output"
                            class="output-image"
                        />
                    </div>
                </Show>

                {/* No image but no error either */}
                <Show when={!props.result?.image_b64 && !props.result?.error}>
                    <div class="output-no-image">
                        <p>No image output. Call <code>cv2.imshow('name', img)</code> to display an image.</p>
                    </div>
                </Show>

                {/* Logs */}
                <Show when={props.result?.logs}>
                    <div class="output-logs">
                        <div class="output-logs-label">Logs</div>
                        <pre class="output-logs-content">{props.result!.logs}</pre>
                    </div>
                </Show>

                {/* Error */}
                <Show when={props.result?.error}>
                    <div class="output-error">
                        <div class="output-error-label">Error</div>
                        <pre class="output-error-content">{props.result!.error}</pre>
                    </div>
                </Show>
            </Show>
        </div>
    );
};

export default OutputPanel;
