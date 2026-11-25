# Fashion Brain frontend (separate project)

This folder is **not** part of the Python serve process.

| Process | Command | Port |
| --- | --- | --- |
| Knowledge graph API | `python -m knowledge_graph.cli serve` | 8001 |
| Fashion Brain API | `python -m fashion_brain.cli serve` | 8000 |
| **This UI** | `npm start` (from `frontend/`) | 5173 |

## Run

```powershell
# terminals 1–2: backends
python -m knowledge_graph.cli serve
python -m fashion_brain.cli serve

# terminal 3: frontend
cd frontend
npm start
```

Open http://127.0.0.1:5173/ (redirects to login) or http://127.0.0.1:5173/login.html  
On the LAN: `http://<this-pc-ip>:5173/` (UI, KG, and Fashion Brain all bind `0.0.0.0` by default).

- User → `/user/`; conversations use `/user/<username>/<chat-id>` and can be reopened directly.
- Admin → `/admin/`

Conversation links are resolved by the UI and loaded through the authenticated API, which continues to enforce conversation ownership. Deep-link refreshes are supported by the `serve.json` rewrite when using `npm start`; Python's built-in static server does not read that rewrite configuration.

The admin sidebar follows the KG Postman **“0 UI wizard (empty → live)”** flow: **KG setup wizard** handles inspect → ontology → mapping → database connection → source → graph → checklist → build/status → query; **Graph overview** shows registered resources and sync/rebuild actions. Legacy PLM-ingest and standalone query panels are not part of this admin UI. The wizard calls the authenticated Fashion Brain admin API, which proxies the required Knowledge Graph endpoints server-to-server; the browser does not need direct access to the KG service or its credentials. Configure the KG URL and any `FASHIONGPT_API_TOKEN` on the backend. Database passwords are not collected by the UI; configure the password environment variable on the KG server.

The Fashion Brain API base URL is set in [`shared/config.js`](shared/config.js) from the page hostname (`http://<same-host>:8000`). Override with `window.FB_API_BASE` if needed.

Without npm (also LAN-reachable):

```powershell
cd frontend
python -m http.server 5173 --bind 0.0.0.0
```

## Layout

```text
frontend/
  login.html          sign in / register
  user/               chat UI
  admin/              KG admin console
  shared/             config.js, api.js, CSS
```
