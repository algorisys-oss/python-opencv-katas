import { Accessor, createContext, createEffect, createSignal, JSX, useContext } from "solid-js";

type Theme = "light" | "dark";

interface ThemeContextType {
    theme: Accessor<Theme>;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType>();

export function ThemeProvider(props: { children: JSX.Element }) {
    // Default to dark mode if no preference is saved
    const [theme, setTheme] = createSignal<Theme>(
        (localStorage.getItem("theme") as Theme) || "dark"
    );

    createEffect(() => {
        const currentTheme = theme();
        localStorage.setItem("theme", currentTheme);
        document.documentElement.setAttribute("data-theme", currentTheme);
    });

    const toggleTheme = () => {
        setTheme((prev) => (prev === "light" ? "dark" : "light"));
    };

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {props.children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error("useTheme must be used within a ThemeProvider");
    }
    return context;
}
