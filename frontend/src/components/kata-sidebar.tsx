/**
 * kata-sidebar.tsx — Left sidebar listing all katas grouped by level.
 * Shows completion checkmarks when user is logged in.
 */

import { Component, For, createMemo, onMount } from "solid-js";
import { A } from "@solidjs/router";
import type { KataListItem } from "../api/client";
import { useTheme } from "../context/ThemeContext";
import AuthDropdown from "./auth-dropdown";

interface Props {
    katas: KataListItem[];
    currentSlug: string;
    completedSlugs?: Set<string>;
}

const LEVEL_ORDER = ["beginner", "intermediate", "advanced", "live"] as const;

const LEVEL_LABELS: Record<string, string> = {
    beginner: "Beginner",
    intermediate: "Intermediate",
    advanced: "Advanced",
    live: "Live Camera",
};

const KataSidebar: Component<Props> = (props) => {
    const { theme, toggleTheme } = useTheme();
    let navRef: HTMLElement | undefined;

    // Scroll the active kata into view on first load only
    onMount(() => {
        requestAnimationFrame(() => {
            navRef?.querySelector(".sidebar-item--active")
                ?.scrollIntoView({ block: "center", behavior: "instant" });
        });
    });

    // Map slug -> serial number (based on sorted index)
    const serialMap = createMemo(() => {
        const map = new Map<string, number>();
        props.katas.forEach((k, i) => {
            map.set(k.slug, i + 1);
        });
        return map;
    });

    const grouped = createMemo(() => {
        const map: Record<string, KataListItem[]> = {};
        for (const kata of props.katas) {
            if (!map[kata.level]) map[kata.level] = [];
            map[kata.level].push(kata);
        }
        return map;
    });

    return (
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-brand">
                    <img class="sidebar-logo-img" src="/logo.svg" alt="Logo" width="22" height="22" />
                    <span class="sidebar-title">OpenCV Playground</span>
                </div>
                <div class="sidebar-header-actions">
                    <AuthDropdown />
                    <button
                        class="btn btn--icon"
                        onClick={toggleTheme}
                        title={`Switch to ${theme() === "light" ? "Dark" : "Light"} Mode`}
                    >
                        {theme() === "light" ? "🌙" : "☀️"}
                    </button>
                </div>
            </div>

            <nav class="sidebar-nav" ref={navRef}>
                <For each={LEVEL_ORDER}>
                    {(level) =>
                        grouped()[level]?.length ? (
                            <div class="sidebar-group">
                                <div class={`sidebar-group-label sidebar-group-label--${level}`}>
                                    {LEVEL_LABELS[level]}
                                </div>
                                <For each={grouped()[level]}>
                                    {(kata) => (
                                        <A
                                            href={`/kata/${kata.slug}`}
                                            class="sidebar-item"
                                            classList={{ "sidebar-item--active": kata.slug === props.currentSlug }}
                                        >
                                            <span class="sidebar-item-title">
                                                {props.completedSlugs?.has(kata.slug) && (
                                                    <span class="sidebar-item-check">✓ </span>
                                                )}
                                                {serialMap().get(kata.slug)}. {kata.title}
                                            </span>
                                            <div class="sidebar-item-concepts">
                                                <For each={kata.concepts.slice(0, 2)}>
                                                    {(c) => <span class="concept-tag">{c}</span>}
                                                </For>
                                            </div>
                                        </A>
                                    )}
                                </For>
                            </div>
                        ) : null
                    }
                </For>
            </nav>
        </aside>
    );
};

export default KataSidebar;
