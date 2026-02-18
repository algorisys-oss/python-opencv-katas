/**
 * demo-panel.tsx — Renders the kata's Markdown body using marked.
 * The full kata content (description, tips, code examples) is written
 * in Markdown and displayed here with syntax highlighting via highlight.js.
 */

import { Component, createMemo } from "solid-js";
import { marked } from "marked";
import type { KataDetail } from "../api/client";

interface Props {
    kata: KataDetail;
}

const DemoPanel: Component<Props> = (props) => {
    const html = createMemo(() => {
        if (!props.kata.body) return "";
        return marked.parse(props.kata.body) as string;
    });

    return (
        <div class="demo-panel">
            <div
                class="kata-markdown"
                innerHTML={html()}
            />
        </div>
    );
};

export default DemoPanel;
