/**
 * api/client.ts — Typed API client for the OpenCV Playground backend.
 * Automatically attaches Authorization header when a token exists.
 */

const BASE = "/api";

export interface KataListItem {
    id: number;
    slug: string;
    title: string;
    level: "beginner" | "intermediate" | "advanced" | "live";
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

export interface ProgressItem {
    kata_id: number;
    kata_slug: string;
    completed_at: string;
}

export interface SavedCode {
    id: number;
    kata_id: number;
    code: string;
    saved_at: string;
}

function authHeaders(): Record<string, string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const token = localStorage.getItem("auth_token");
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    return headers;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE}${path}`, {
        headers: authHeaders(),
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

    executeCode: (code: string, local: boolean = false) =>
        request<ExecuteResult>("/execute", {
            method: "POST",
            body: JSON.stringify({ code, local }),
        }),

    stopExecution: () =>
        request<{ stopped: boolean; message: string }>("/execute/stop", {
            method: "POST",
        }),

    // Auth
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

    // Progress
    getProgress: () => request<ProgressItem[]>("/me/progress"),

    markComplete: (slug: string) =>
        request<{ status: string }>(`/katas/${slug}/complete`, { method: "POST" }),

    unmarkComplete: (slug: string) =>
        request<{ status: string }>(`/katas/${slug}/complete`, { method: "DELETE" }),

    // Code saving
    saveCode: (slug: string, code: string) =>
        request<SavedCode>(`/katas/${slug}/save`, {
            method: "POST",
            body: JSON.stringify({ code }),
        }),

    getSavedCode: (slug: string) =>
        request<SavedCode | null>(`/katas/${slug}/saved`),
};
