/** Frontend project config. Points at the Fashion Brain API (separate process).
 *
 * When opened from another device on the LAN (http://<pc-ip>:5173), the API base
 * uses that same hostname on :8000. Override anytime with window.FB_API_BASE.
 */
(function (global) {
  if (global.FB_API_BASE) return;
  var host =
    typeof global.location !== "undefined" && global.location.hostname
      ? global.location.hostname
      : "127.0.0.1";
  var protocol =
    typeof global.location !== "undefined" && global.location.protocol
      ? global.location.protocol
      : "http:";
  // file:// or odd protocols → default to local Fashion Brain.
  if (protocol !== "http:" && protocol !== "https:") {
    protocol = "http:";
    host = "127.0.0.1";
  }
  global.FB_API_BASE = protocol + "//" + host + ":8000";
})(typeof window !== "undefined" ? window : globalThis);
