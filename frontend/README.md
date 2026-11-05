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

Open http://127.0.0.1:5173/login.html

- User → `/user/`
- Admin → `/admin/`

Admin **Scraping wizard** walks extract → chunk → propose → queue → approve → commit. Each step shows input and output; use **Next** after a successful run. Commit can target an existing ontology/graph or create new ones linked to master.

API base URL is set in [`shared/config.js`](shared/config.js) (`http://127.0.0.1:8000`). Change it if Fashion Brain listens elsewhere.

Without npm:

```powershell
cd frontend
python -m http.server 5173
```

## Layout

```text
frontend/
  login.html          sign in / register
  user/               chat UI
  admin/              KG admin console
  shared/             config.js, api.js, CSS
```
