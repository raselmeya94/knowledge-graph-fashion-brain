/** Fashion Brain API helper. Frontend is a separate origin; API base comes from config.js. */
(function (global) {
  const KEYS = {
    access: "fb.accessToken",
    refresh: "fb.refreshToken",
    user: "fb.user",
  };

  function apiBase() {
    return String(global.FB_API_BASE || "http://127.0.0.1:8000").replace(/\/$/, "");
  }

  function getUser() {
    try {
      return JSON.parse(localStorage.getItem(KEYS.user) || "null");
    } catch {
      return null;
    }
  }

  function getAccess() {
    return localStorage.getItem(KEYS.access) || "";
  }

  function getRefresh() {
    return localStorage.getItem(KEYS.refresh) || "";
  }

  function saveSession(payload) {
    if (payload.access_token) localStorage.setItem(KEYS.access, payload.access_token);
    if (payload.refresh_token) localStorage.setItem(KEYS.refresh, payload.refresh_token);
    if (payload.user) localStorage.setItem(KEYS.user, JSON.stringify(payload.user));
  }

  function clearSession() {
    localStorage.removeItem(KEYS.access);
    localStorage.removeItem(KEYS.refresh);
    localStorage.removeItem(KEYS.user);
  }

  async function refreshAccess() {
    const refresh = getRefresh();
    if (!refresh) throw new Error("Not signed in.");
    const res = await fetch(apiBase() + "/v1/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      clearSession();
      throw new Error(body.message || "Session expired.");
    }
    saveSession(body);
    return body.access_token;
  }

  async function request(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    const token = getAccess();
    if (token && !options.noAuth) headers.Authorization = "Bearer " + token;
    if (options.clearance) headers["X-Clearance"] = options.clearance;
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    let res = await fetch(apiBase() + path, Object.assign({}, options, { headers }));
    if (res.status === 401 && !options.noAuth && getRefresh()) {
      await refreshAccess();
      headers.Authorization = "Bearer " + getAccess();
      res = await fetch(apiBase() + path, Object.assign({}, options, { headers }));
    }
    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    if (!res.ok) {
      const err = new Error((data && (data.message || data.error)) || res.statusText || "Request failed");
      err.status = res.status;
      err.body = data;
      throw err;
    }
    return data;
  }

  function requireAuth(expectedRole) {
    const user = getUser();
    const token = getAccess();
    if (!user || !token) {
      location.href = "/login.html";
      return null;
    }
    if (expectedRole === "admin" && user.role !== "admin") {
      location.href = "/user/";
      return null;
    }
    if (expectedRole === "user" && user.role === "admin" && location.pathname.indexOf("/admin") === 0) {
      /* admin viewing admin is fine */
    }
    return user;
  }

  function homeFor(user) {
    return user && user.role === "admin" ? "/admin/" : "/user/";
  }

  global.FB = {
    KEYS,
    apiBase,
    getUser,
    getAccess,
    getRefresh,
    saveSession,
    clearSession,
    request,
    requireAuth,
    homeFor,
    login: (email, password) =>
      request("/v1/auth/login", {
        method: "POST",
        noAuth: true,
        body: JSON.stringify({ email, password }),
      }).then((data) => {
        saveSession(data);
        return data;
      }),
    register: (payload) =>
      request("/v1/auth/register", {
        method: "POST",
        noAuth: true,
        body: JSON.stringify(payload),
      }),
    logout: async () => {
      try {
        await request("/v1/auth/logout", {
          method: "POST",
          body: JSON.stringify({ refresh_token: getRefresh() || null }),
        });
      } catch {
        /* ignore */
      }
      clearSession();
      location.href = "/login.html";
    },
  };
})(window);
