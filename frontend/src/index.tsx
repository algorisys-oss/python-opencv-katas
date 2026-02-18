/* Render root */
import { render } from "solid-js/web";
import { Router } from "@solidjs/router";
import App from "./App";
import { ThemeProvider } from "./context/ThemeContext";
import "./index.css";
import "./components.css";

const root = document.getElementById("root");
if (!root) throw new Error("No #root element found");

render(() => <ThemeProvider><Router><App /></Router></ThemeProvider>, root);
