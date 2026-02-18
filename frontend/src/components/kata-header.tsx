/**
 * kata-header.tsx — Displays kata title, level badge, and concept tags.
 */

import { Component, For } from "solid-js";
import type { KataDetail } from "../api/client";

interface Props {
    kata: KataDetail;
}

const KataHeader: Component<Props> = (props) => {
    return (
        <header class="kata-header">
            <div class="kata-header-top">
                <h1 class="kata-title">{props.kata.title}</h1>
                <span class={`level-badge level-badge--${props.kata.level}`}>
                    {props.kata.level}
                </span>
            </div>
            <div class="kata-concepts">
                <For each={props.kata.concepts}>
                    {(concept) => (
                        <span class="concept-chip">
                            <code>{concept}</code>
                        </span>
                    )}
                </For>
            </div>
        </header>
    );
};

export default KataHeader;
