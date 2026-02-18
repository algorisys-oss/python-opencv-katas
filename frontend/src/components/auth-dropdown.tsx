/**
 * auth-dropdown.tsx — Dropdown for login/register from the sidebar header.
 * Shows a user icon when logged out, email + logout when logged in.
 * Uses fixed positioning to avoid sidebar overflow clipping.
 */

import { Component, createSignal, Show } from "solid-js";
import { useAuth } from "../context/AuthContext";

type FormMode = "login" | "register";

const AuthDropdown: Component = () => {
    const { user, login, register, logout } = useAuth();
    const [open, setOpen] = createSignal(false);
    const [mode, setMode] = createSignal<FormMode>("login");
    const [email, setEmail] = createSignal("");
    const [password, setPassword] = createSignal("");
    const [error, setError] = createSignal("");
    const [loading, setLoading] = createSignal(false);
    const [panelPos, setPanelPos] = createSignal({ top: 0, left: 0 });

    let btnRef: HTMLButtonElement | undefined;

    const toggle = () => {
        if (!open() && btnRef) {
            const rect = btnRef.getBoundingClientRect();
            setPanelPos({ top: rect.bottom + 6, left: rect.left });
        }
        setOpen(!open());
    };

    const handleSubmit = async (e: Event) => {
        e.preventDefault();
        setError("");
        setLoading(true);
        try {
            if (mode() === "login") {
                await login(email(), password());
            } else {
                await register(email(), password());
            }
            setOpen(false);
            setEmail("");
            setPassword("");
        } catch (err: any) {
            setError(err.message || "Something went wrong");
        } finally {
            setLoading(false);
        }
    };

    const handleLogout = () => {
        logout();
        setOpen(false);
    };

    return (
        <div class="auth-dropdown">
            <button
                ref={btnRef}
                class="btn btn--icon"
                onClick={toggle}
                title={user() ? user()!.email : "Sign in"}
            >
                {user() ? "👤" : "🔑"}
            </button>

            <Show when={open()}>
                <div class="auth-dropdown-overlay" onClick={() => setOpen(false)} />
                <div
                    class="auth-dropdown-panel"
                    style={{ top: `${panelPos().top}px`, left: `${panelPos().left}px` }}
                >
                    <Show
                        when={user()}
                        fallback={
                            <>
                                <div class="auth-form-tabs">
                                    <button
                                        class="auth-form-tab"
                                        classList={{ "auth-form-tab--active": mode() === "login" }}
                                        onClick={() => { setMode("login"); setError(""); }}
                                    >
                                        Sign In
                                    </button>
                                    <button
                                        class="auth-form-tab"
                                        classList={{ "auth-form-tab--active": mode() === "register" }}
                                        onClick={() => { setMode("register"); setError(""); }}
                                    >
                                        Register
                                    </button>
                                </div>
                                <form class="auth-form" onSubmit={handleSubmit}>
                                    <input
                                        class="auth-input"
                                        type="email"
                                        placeholder="Email"
                                        value={email()}
                                        onInput={(e) => setEmail(e.currentTarget.value)}
                                        required
                                    />
                                    <input
                                        class="auth-input"
                                        type="password"
                                        placeholder="Password"
                                        value={password()}
                                        onInput={(e) => setPassword(e.currentTarget.value)}
                                        required
                                        minLength={6}
                                    />
                                    <Show when={error()}>
                                        <div class="auth-error">{error()}</div>
                                    </Show>
                                    <button
                                        class="btn btn--primary auth-submit"
                                        type="submit"
                                        disabled={loading()}
                                    >
                                        {loading() ? "..." : mode() === "login" ? "Sign In" : "Create Account"}
                                    </button>
                                </form>
                            </>
                        }
                    >
                        <div class="auth-user-info">
                            <span class="auth-user-email">{user()!.email}</span>
                            <button class="btn btn--ghost auth-logout-btn" onClick={handleLogout}>
                                Sign Out
                            </button>
                        </div>
                    </Show>
                </div>
            </Show>
        </div>
    );
};

export default AuthDropdown;
