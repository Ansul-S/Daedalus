export const THEME_KEY = "daedalus:theme";

/** Runs in <head> before the first paint, so a saved theme never flashes the other one first.
 * Without a saved choice the page follows the system through CSS alone. */
export const THEME_SCRIPT = `try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
