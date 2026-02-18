/**
 * App.tsx — Root application with routing and layout.
 * Fetches user progress when logged in.
 */

import { Component, createResource, createSignal, createEffect, Show } from "solid-js";
import { Route, useLocation, useNavigate } from "@solidjs/router";
import { api } from "./api/client";
import { useAuth } from "./context/AuthContext";
import KataSidebar from "./components/kata-sidebar";
import KataPage from "./pages/kata-page";

// Layout wraps every page with the sidebar
const Layout: Component<{ children?: any }> = (props) => {
    const [katas] = createResource(api.getKatas);
    const { user } = useAuth();
    const navigate = useNavigate();

    // Track completed slugs — refetch when user changes
    const [completedSlugs, setCompletedSlugs] = createSignal<Set<string>>(new Set());

    const fetchProgress = async () => {
        if (!user()) {
            setCompletedSlugs(new Set<string>());
            return;
        }
        try {
            const progress = await api.getProgress();
            setCompletedSlugs(new Set<string>(progress.map((p) => p.kata_slug)));
        } catch {
            setCompletedSlugs(new Set<string>());
        }
    };

    // Refetch progress when user logs in/out
    createEffect(() => {
        user(); // track
        fetchProgress();
    });

    // Expose refreshProgress for child components
    (window as any).__refreshProgress = fetchProgress;

    // Redirect root to first kata
    const handleRootRedirect = () => {
        const list = katas();
        if (list && list.length > 0) {
            navigate(`/kata/${list[0].slug}`, { replace: true });
        }
    };

    // Get current slug from URL (reactive via router)
    const location = useLocation();
    const currentSlug = () => {
        const parts = location.pathname.split("/");
        return parts[parts.length - 1] ?? "";
    };

    return (
        <div class="app-layout">
            <Show when={katas()} fallback={<div class="sidebar-loading">Loading…</div>}>
                {(list) => {
                    // Auto-redirect from root
                    if (location.pathname === "/") handleRootRedirect();
                    return (
                        <KataSidebar
                            katas={list()}
                            currentSlug={currentSlug()}
                            completedSlugs={completedSlugs()}
                        />
                    );
                }}
            </Show>
            <main class="app-main">{props.children}</main>
        </div>
    );
};

const App: Component = () => {
    return (
        <>
            <Route path="/" component={Layout}>
                <Route path="/" component={() => <div class="kata-loading">Loading katas…</div>} />
                <Route path="/kata/:slug" component={KataPage} />
            </Route>
        </>
    );
};

export default App;
