/**
 * App.tsx — Root application with routing and layout.
 */

import { Component, createResource, Show } from "solid-js";
import { Route, useLocation, useNavigate } from "@solidjs/router";
import { api } from "./api/client";
import KataSidebar from "./components/kata-sidebar";
import KataPage from "./pages/kata-page";

// Layout wraps every page with the sidebar
const Layout: Component<{ children?: any }> = (props) => {
    const [katas] = createResource(api.getKatas);
    const navigate = useNavigate();

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
                    return <KataSidebar katas={list()} currentSlug={currentSlug()} />;
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
