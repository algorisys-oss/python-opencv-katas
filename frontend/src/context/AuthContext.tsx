/**
 * AuthContext.tsx — Optional authentication context.
 * Stores JWT in localStorage, decodes email from token payload.
 * Anonymous users get null — all features work without login.
 */

import { Accessor, createContext, createSignal, JSX, useContext } from "solid-js";
import { api } from "../api/client";

export interface AuthUser {
    email: string;
}

interface AuthContextType {
    user: Accessor<AuthUser | null>;
    token: Accessor<string | null>;
    login: (email: string, password: string) => Promise<void>;
    register: (email: string, password: string) => Promise<void>;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType>();

function decodePayload(token: string): { email: string; exp: number } | null {
    try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        return { email: payload.email, exp: payload.exp };
    } catch {
        return null;
    }
}

function isExpired(exp: number): boolean {
    return Date.now() / 1000 > exp;
}

export function AuthProvider(props: { children: JSX.Element }) {
    // Restore token from localStorage on init
    const stored = localStorage.getItem("auth_token");
    let initialUser: AuthUser | null = null;
    let initialToken: string | null = null;

    if (stored) {
        const payload = decodePayload(stored);
        if (payload && !isExpired(payload.exp)) {
            initialUser = { email: payload.email };
            initialToken = stored;
        } else {
            localStorage.removeItem("auth_token");
        }
    }

    const [user, setUser] = createSignal<AuthUser | null>(initialUser);
    const [token, setToken] = createSignal<string | null>(initialToken);

    const handleToken = (accessToken: string) => {
        const payload = decodePayload(accessToken);
        if (!payload) throw new Error("Invalid token received");
        localStorage.setItem("auth_token", accessToken);
        setToken(accessToken);
        setUser({ email: payload.email });
    };

    const login = async (email: string, password: string) => {
        const res = await api.login(email, password);
        handleToken(res.access_token);
    };

    const register = async (email: string, password: string) => {
        const res = await api.register(email, password);
        handleToken(res.access_token);
    };

    const logout = () => {
        localStorage.removeItem("auth_token");
        setToken(null);
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, token, login, register, logout }}>
            {props.children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
