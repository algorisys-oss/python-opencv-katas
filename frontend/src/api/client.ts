/**
 * api/client.ts — Typed API client for the OpenCV Playground backend.
 */

const BASE = "/api";

export interface KataListItem {
    id: number;
    slug: string;
    title: string;
    level: "beginner" | "intermediate" | "advanced";
    concepts: string[];
}

export interface KataDetail extends KataListItem {
    body: string;              // Full Markdown body
    prerequisites: string[];
    starter_code: string;
    demo_controls: DemoControl[];
}

export interface DemoControl {
    type: "slider" | "toggle" | "dropdown";
    label: string;
    param: string;
    min?: number;
    max?: number;
    step?: number;
    default?: number | boolean | string;
    options?: string[];
}

export interface ExecuteResult {
    image_b64: string | null;
    logs: string;
    error: string;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
    }
    return res.json();
}

export const api = {
    getKatas: () => request<KataListItem[]>("/katas"),

    getKata: (slug: string) => request<KataDetail>(`/katas/${slug}`),

    executeCode: (code: string) =>
        request<ExecuteResult>("/execute", {
            method: "POST",
            body: JSON.stringify({ code }),
        }),

    login: (email: string, password: string) =>
        request<{ access_token: string; token_type: string }>("/auth/login", {
            method: "POST",
            body: JSON.stringify({ email, password }),
        }),

    register: (email: string, password: string) =>
        request<{ access_token: string; token_type: string }>("/auth/register", {
            method: "POST",
            body: JSON.stringify({ email, password }),
        }),
};
