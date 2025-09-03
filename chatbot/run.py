#!/usr/bin/env python3
import os
from pathlib import Path
from flask import Flask, request, send_from_directory, Response, jsonify
import requests
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

REALTIME_SDP_ENDPOINT = "https://api.openai.com/v1/realtime"
REGISTRY_DIR = Path(os.environ.get("HOUND_REGISTRY_DIR", os.path.expanduser("~/.local/state/hound/instances")))
ACTIVE_PROJECT_FILE = ROOT / ".active_project"


def _get_openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def create_app():
    app = Flask(__name__, static_folder=str(STATIC_DIR))

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/static/<path:path>")
    def static_files(path: str):
        return send_from_directory(app.static_folder, path)

    @app.get("/health")
    def health():
        return jsonify({
            "ok": True,
            "has_api_key": bool(_get_openai_api_key()),
            "active_project": (_read_active_project() or "")
        })

    # ---- Simple server-side context for active project ----
    def _read_active_project() -> str | None:
        try:
            if ACTIVE_PROJECT_FILE.exists():
                v = ACTIVE_PROJECT_FILE.read_text(encoding="utf-8").strip()
                return v or None
        except Exception:
            return None
        return None

    def _write_active_project(pid: str | None):
        try:
            if not pid:
                if ACTIVE_PROJECT_FILE.exists():
                    ACTIVE_PROJECT_FILE.unlink()
                return
            ACTIVE_PROJECT_FILE.write_text(pid, encoding="utf-8")
        except Exception:
            pass

    @app.get("/api/context")
    def get_context():
        return jsonify({"project_id": _read_active_project()})

    @app.post("/api/context")
    def set_context():
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception:
            data = {}
        pid = (data.get("project_id") or "").strip()
        _write_active_project(pid or None)
        return jsonify({"ok": True, "project_id": _read_active_project()})

    @app.post("/webrtc/offer")
    def webrtc_offer():
        model = request.args.get("model", "gpt-4o-realtime-preview-2024-12-17")
        voice = request.args.get("voice", "verse")
        api_key = _get_openai_api_key()
        if not api_key:
            return jsonify({"error": "OPENAI_API_KEY not set"}), 500
        sdp_offer = request.get_data(as_text=True)
        if not sdp_offer:
            return jsonify({"error": "Missing SDP offer body"}), 400

        url = f"{REALTIME_SDP_ENDPOINT}?model={model}&voice={voice}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/sdp",
            "Accept": "application/sdp",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            r = requests.post(url, headers=headers, data=sdp_offer, timeout=15)
        except Exception as e:
            return jsonify({"error": "Upstream request error", "detail": str(e)}), 502
        if r.status_code >= 400:
            return jsonify({"error": "Upstream error", "status": r.status_code, "body": r.text}), 502
        return Response(r.text, status=200, mimetype="application/sdp")

    # ---- Hound instance registry APIs ----
    def _read_instance(file: Path) -> dict | None:
        try:
            data = json.loads(file.read_text())
            data["_id"] = file.stem
            return data
        except Exception:
            return None

    @app.get("/api/instances")
    def api_instances():
        out = []
        if REGISTRY_DIR.exists():
            for f in sorted(REGISTRY_DIR.glob("*.json")):
                inst = _read_instance(f)
                if inst:
                    out.append({
                        "id": inst.get("_id", f.stem),
                        "pid": inst.get("pid"),
                        "project_id": inst.get("project_id"),
                        "started_at": inst.get("started_at"),
                        "telemetry": inst.get("telemetry", {}),
                    })
        return jsonify({"instances": out})

    @app.get("/api/instance/recent")
    def api_instance_recent():
        """Return last N telemetry events for an instance as JSON (proxy)."""
        inst_id = request.args.get("id", "").strip()
        try:
            limit = int(request.args.get("limit", "40"))
        except Exception:
            limit = 40
        if not inst_id:
            return jsonify({"error": "missing id"}), 400
        inst_file = None
        if REGISTRY_DIR.exists():
            for f in REGISTRY_DIR.glob("*.json"):
                if f.stem == inst_id:
                    inst_file = f
                    break
        if not inst_file:
            return jsonify({"error": "instance not found"}), 404
        inst = _read_instance(inst_file) or {}
        tel = inst.get("telemetry", {}) or {}
        ctrl = tel.get("control_url")
        token = tel.get("token")
        if not ctrl:
            return jsonify({"error": "no control url"}), 400
        try:
            r = requests.get(f"{ctrl}/recent", params={"limit": limit, **({"token": token} if token else {})}, timeout=5)
            if r.status_code >= 400:
                return jsonify({"error": f"upstream {r.status_code}"}), 502
            return Response(r.content, status=200, mimetype="application/json")
        except Exception as e:
            return jsonify({"error": str(e)}), 502

    @app.get("/api/instance/status")
    def api_instance_status():
        """SSE status stream for a given instance id.
        If the instance advertises a telemetry.sse_url, proxy it.
        Otherwise, emit a minimal heartbeat with unknown status.
        """
        inst_id = request.args.get("id", "").strip()
        if not inst_id:
            return jsonify({"error": "missing id"}), 400
        inst_file = None
        if REGISTRY_DIR.exists():
            for f in REGISTRY_DIR.glob("*.json"):
                if f.stem == inst_id:
                    inst_file = f
                    break
        if not inst_file:
            return jsonify({"error": "instance not found"}), 404
        inst = _read_instance(inst_file) or {}
        tel = inst.get("telemetry", {}) or {}
        sse_url = tel.get("sse_url")
        token = tel.get("token")

        def _proxy():
            if sse_url:
                try:
                    params = {}
                    if token:
                        params["token"] = token
                    with requests.get(sse_url, params=params, stream=True, timeout=10) as r:
                        if r.status_code >= 400:
                            yield f"data: {{\"status\":\"error\",\"message\":\"upstream {r.status_code}\"}}\n\n"
                        else:
                            for chunk in r.iter_lines(decode_unicode=True):
                                if chunk is None:
                                    continue
                                if chunk.startswith("data:"):
                                    yield chunk + "\n\n"
                                else:
                                    yield f"data: {chunk}\n\n"
                            return
                except Exception as e:
                    yield f"data: {{\"status\":\"error\",\"message\":\"proxy failed: {str(e)}\"}}\n\n"
            # Fallback heartbeat
            pid = inst.get("pid")
            proj = inst.get("project_id")
            while True:
                payload = {"status": "unknown", "project_id": proj, "pid": pid, "ts": time.time()}
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1.0)
        return Response(_proxy(), mimetype="text/event-stream")

    def _hound_cli(*parts: str) -> subprocess.CompletedProcess:
        hound_cli = ROOT.parent / "hound.py"
        cmd = [os.environ.get("PYTHON", "python"), str(hound_cli), *parts]
        return subprocess.run(cmd, cwd=str(ROOT.parent), capture_output=True, text=True)

    def _project_path(project_id: str) -> Path | None:
        """Resolve a project id or filesystem path to a project directory.
        Accepts either a known project name or an absolute/relative path.
        """
        try:
            # If it's a path that exists, use it directly
            p = Path(project_id)
            if p.exists():
                return p
        except Exception:
            pass
        # Otherwise, try to resolve by project name via CLI
        try:
            proc = _hound_cli("project", "path", project_id)
            if proc.returncode == 0:
                p = Path(proc.stdout.strip())
                return p if p.exists() else None
        except Exception:
            pass
        return None

    def _read_latest_session(proj: Path) -> dict:
        sessions_dir = proj / "sessions"
        session_file = None
        if sessions_dir.exists():
            jsons = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            session_file = jsons[0] if jsons else None
        if not session_file:
            return {}
        try:
            with session_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            try:
                return json.loads(session_file.read_text())
            except Exception:
                return {}

    def _resolve_graphs(proj: Path) -> dict:
        """Return mapping of graph name -> path string.
        Uses graphs/knowledge_graphs.json if present; else scans for graph_*.json.
        """
        graphs_dir = proj / 'graphs'
        mapping = {}
        try:
            kg = graphs_dir / 'knowledge_graphs.json'
            if kg.exists():
                meta = json.loads(kg.read_text())
                g = meta.get('graphs') or {}
                # Normalize to absolute paths
                for k, v in g.items():
                    mapping[k] = str((Path(v) if os.path.isabs(v) else (graphs_dir / v)).resolve())
                return mapping
        except Exception:
            mapping = {}
        # Fallback: scan files
        try:
            for f in graphs_dir.glob('graph_*.json'):
                name = f.stem.replace('graph_', '')
                mapping[name] = str(f.resolve())
        except Exception:
            pass
        return mapping

    def _resolve_system_graph(proj: Path) -> str | None:
        gmap = _resolve_graphs(proj)
        for key in ('SystemOverview', 'SystemArchitecture', 'Architecture', 'Overview'):
            if key in gmap:
                return gmap[key]
        # Fallback: first available graph
        return next(iter(gmap.values()), None)

    _cards_cache = {}
    def _cards_index(proj: Path):
        """Return tuple (id_to_rel, rel_to_id, rel_to_snippets).
        rel_to_snippets maps relpath to list of small code snippets/cards if available.
        """
        key = str(proj)
        if key in _cards_cache:
            return _cards_cache[key]
        id_to_rel = {}
        rel_to_id = {}
        rel_to_snips = {}
        mdir = proj / 'manifest'
        try:
            cj = mdir / 'cards.jsonl'
            if cj.exists():
                with cj.open('r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        cid = str(obj.get('id') or '')
                        rel = obj.get('relpath') or ''
                        if rel:
                            id_to_rel[cid] = rel
                            rel_to_id.setdefault(rel, cid)
                            content = obj.get('content') or obj.get('snippet') or ''
                            if content:
                                rel_to_snips.setdefault(rel, []).append(content)
        except Exception:
            pass
        _cards_cache[key] = (id_to_rel, rel_to_id, rel_to_snips)
        return _cards_cache[key]
        try:
            return json.loads(session_file.read_text())
        except Exception:
            return {}

    @app.get("/api/stream/audit")
    def stream_audit():
        project_id = request.args.get("project", "").strip()
        if not project_id:
            return jsonify({"error": "missing project"}), 400
        proj = _project_path(project_id)
        if not proj:
            return jsonify({"error": f"project not found: {project_id}"}), 404
        # Prefer .hound_debug/agent.log; else show a placeholder
        log_path = proj / ".hound_debug" / "agent.log"
        if not log_path.exists():
            def _gen_empty():
                # Emit a one-off info line, then heartbeat to keep EventSource alive
                yield f"data: [info] No debug log found at {log_path}\n\n"
                while True:
                    yield f"data: {{\"status\":\"idle\",\"ts\":{time.time()} }}\n\n"
                    time.sleep(1.0)
            return Response(_gen_empty(), mimetype="text/event-stream")

        def _gen():
            try:
                with log_path.open('r', encoding='utf-8', errors='ignore') as f:
                    # Replay the last ~80 lines for context
                    try:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        to_read = min(size, 8192)
                        f.seek(size - to_read)
                        chunk = f.read()
                        lines = chunk.splitlines()[-80:]
                        for ln in lines:
                            yield f"data: {ln}\n\n"
                    except Exception:
                        pass
                    # Now follow the file
                    f.seek(0, os.SEEK_END)
                    while True:
                        where = f.tell()
                        line = f.readline()
                        if not line:
                            time.sleep(0.5)
                            f.seek(where)
                        else:
                            # SSE frame
                            yield f"data: {line.rstrip()}\n\n"
            except GeneratorExit:
                return
            except Exception as e:
                yield f"data: [error] {str(e)}\n\n"
        return Response(_gen(), mimetype="text/event-stream")

    # Simple function-calling wired tools
    @app.post("/api/tool/<name>")
    def tool_call(name: str):
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {}
        # Optional project override via payload; else use active context
        project_id = (payload.get("project_id") or _read_active_project() or "").strip()
        # Helper: resolve project dir once for tools needing it
        proj = _project_path(project_id) if project_id else None
        if name == "get_hound_status":
            # Return a short status using session tracker files if present
            try:
                info: dict = {"ok": True, "project_id": project_id}
                if not project_id:
                    info.update({"summary": "No active project selected"})
                    return jsonify(info)
                if not proj:
                    return jsonify({"ok": False, "error": f"project not found: {project_id}"}), 404
                sessions_dir = proj / "sessions"
                # Pick latest session json by mtime
                session_file = None
                if sessions_dir.exists():
                    jsons = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    session_file = jsons[0] if jsons else None
                status = "idle"
                coverage = {"nodes": {"visited": 0, "total": 0, "percent": 0.0},
                            "cards": {"visited": 0, "total": 0, "percent": 0.0}}
                token_total = 0
                calls = 0
                if session_file:
                    try:
                        data = json.loads(session_file.read_text())
                        status = data.get("status", status)
                        coverage = data.get("coverage", coverage)
                        tu = (data.get("token_usage") or {}).get("total_usage", {})
                        token_total = int(tu.get("total_tokens", 0))
                        calls = int(tu.get("call_count", 0))
                    except Exception:
                        pass
                info.update({
                    "status": status,
                    "coverage": coverage,
                    "token_usage": {"total_tokens": token_total, "call_count": calls},
                    "summary": f"status={status}, coverage={coverage.get('nodes',{}).get('percent',0)}% nodes / {coverage.get('cards',{}).get('percent',0)}% cards, tokens={token_total:,}"
                })
                return jsonify(info)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "enqueue_steering":
            # Append a steering note to the project's .hound/steering.jsonl
            text = (payload.get("text") or "").strip()
            if not text:
                return jsonify({"ok": False, "error": "missing text"}), 400
            if not project_id:
                return jsonify({"ok": False, "error": "no active project"}), 400
            if not proj:
                return jsonify({"ok": False, "error": f"project not found: {project_id}"}), 404
            try:
                steer_file = proj / ".hound" / "steering.jsonl"
                steer_file.parent.mkdir(parents=True, exist_ok=True)
                rec = {"ts": time.time(), "source": "chatbot", "text": text}
                with steer_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                return jsonify({"ok": True, "accepted": True})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "set_emotion":
            # No-op backend; allow the UI to update immediately. Return ok for function_call_output pairing.
            val = (payload.get('value') or '').strip()
            if val not in ('neutral','concentrated','amused','shy'):
                return jsonify({"ok": False, "error": "invalid value"}), 400
            return jsonify({"ok": True})
        if name == "human_status":
            # Return a human-friendly status summary without raw node/card/token numbers
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            try:
                sess = _read_latest_session(proj)
                # Coverage percent by cards
                cov = (sess.get('coverage') or {}).get('cards', {})
                cov_pct = cov.get('percent')
                try:
                    cov_pct = int(round(float(cov_pct))) if cov_pct is not None else None
                except Exception:
                    cov_pct = None
                # Current goal
                planning = sess.get('planning_history') or []
                inv = sess.get('investigations') or []
                current_goal = None
                if planning:
                    last = planning[-1]
                    items = last.get('items') or []
                    done = { (r.get('goal') or '') for r in inv }
                    for it in items:
                        if isinstance(it, dict):
                            g = (it.get('goal') or it.get('description') or it.get('tool_name') or '').strip()
                            if g and g not in done:
                                current_goal = g
                                break
                if not current_goal and inv:
                    current_goal = (inv[-1].get('goal') or '').strip() or None
                # Top hypothesis (non-rejected, highest confidence)
                top = None
                hyp_file = proj / 'hypotheses.json'
                if hyp_file.exists():
                    try:
                        data = json.loads(hyp_file.read_text())
                        hyps = list((data.get('hypotheses') or {}).values())
                        hyps = [h for h in hyps if h.get('status') != 'rejected']
                        hyps.sort(key=lambda h: float(h.get('confidence', 0.0)), reverse=True)
                        if hyps:
                            h = hyps[0]
                            # Shorten description for summary
                            desc = (h.get('description') or '')
                            if len(desc) > 160:
                                desc = desc[:157] + '...'
                            top = {
                                'vulnerability_type': h.get('vulnerability_type') or 'issue',
                                'description': desc,
                                'confidence': h.get('confidence')
                            }
                    except Exception:
                        top = None
                # Compose human summary (avoid raw counts)
                parts = []
                if cov_pct is not None:
                    parts.append(f"We have examined about {cov_pct}% of the codebase so far.")
                else:
                    parts.append("We’re early in the review and still mapping key areas.")
                if current_goal:
                    parts.append(f"Right now we’re digging into: {current_goal}.")
                if top:
                    parts.append(f"Most promising lead: {top['vulnerability_type']} — {top['description']}")
                summary = " ".join(parts)
                return jsonify({
                    "ok": True,
                    "code_coverage_percent": cov_pct,
                    "current_goal": current_goal,
                    "top_hypothesis": top,
                    "summary": summary
                })
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "list_plan":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            sess = _read_latest_session(proj)
            planning = sess.get('planning_history') or []
            items = []
            if planning:
                last = planning[-1]
                cand = last.get('items') or []
                # Completed investigations by goal
                inv = sess.get('investigations') or []
                completed = { (r.get('goal') or '') for r in inv }
                for it in cand:
                    if not isinstance(it, dict):
                        continue
                    goal = it.get('goal') or it.get('description') or it.get('tool_name') or 'Unknown'
                    items.append({
                        'goal': goal,
                        'priority': it.get('priority'),
                        'focus_areas': it.get('focus_areas', []),
                        'done': goal in completed
                    })
            return jsonify({"ok": True, "plan": items})
        if name == "get_current_activity":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            sess = _read_latest_session(proj)
            planning = sess.get('planning_history') or []
            inv = sess.get('investigations') or []
            current_goal = None
            if planning:
                last = planning[-1]
                cand = last.get('items') or []
                done = { (r.get('goal') or '') for r in inv }
                for it in cand:
                    if isinstance(it, dict):
                        g = (it.get('goal') or it.get('description') or it.get('tool_name') or '').strip()
                        if g and g not in done:
                            current_goal = g
                            break
            if not current_goal and inv:
                current_goal = (inv[-1].get('goal') or '').strip() or None
            return jsonify({"ok": True, "current_goal": current_goal, "session_status": sess.get('status', 'unknown')})
        if name == "list_hypotheses":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            try:
                limit = int(payload.get('limit') or 5)
            except Exception:
                limit = 5
            hyp_file = proj / 'hypotheses.json'
            out = []
            if hyp_file.exists():
                try:
                    data = json.loads(hyp_file.read_text())
                    hyps = list((data.get('hypotheses') or {}).values())
                    # Filter
                    status = payload.get('status')
                    if status:
                        hyps = [h for h in hyps if (h.get('status') == status)]
                    # Sort by confidence desc
                    hyps.sort(key=lambda h: float(h.get('confidence', 0.0)), reverse=True)
                    for h in hyps[:limit]:
                        out.append({
                            'id': h.get('id'),
                            'node_id': h.get('node_id'),
                            'vulnerability_type': h.get('vulnerability_type'),
                            'description': h.get('description'),
                            'confidence': h.get('confidence'),
                            'status': h.get('status')
                        })
                except Exception:
                    pass
            return jsonify({"ok": True, "hypotheses": out})
        if name == "get_system_overview":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            try:
                limit = int(payload.get('limit') or 12)
            except Exception:
                limit = 12
            # Resolve a system-level graph
            gpath = _resolve_system_graph(proj)
            if not gpath:
                return jsonify({"ok": False, "error": "system overview graph not found"}), 404
            try:
                g = json.loads(Path(gpath).read_text())
                nodes = g.get('nodes', []) or g.get('data', {}).get('nodes', [])
                edges = g.get('edges', []) or g.get('data', {}).get('edges', [])
                # Compute degree
                deg = {}
                for e in edges:
                    s = str(e.get('source'))
                    t = str(e.get('target'))
                    if s:
                        deg[s] = deg.get(s, 0) + 1
                    if t:
                        deg[t] = deg.get(t, 0) + 1
                lite_nodes = []
                for n in nodes:
                    nid = str(n.get('id'))
                    lite_nodes.append({
                        'id': nid,
                        'label': n.get('label') or nid,
                        'type': n.get('type'),
                        'degree': deg.get(nid, 0)
                    })
                # Top by degree
                top = sorted(lite_nodes, key=lambda x: x.get('degree', 0), reverse=True)[:limit]
                return jsonify({
                    'ok': True,
                    'graph': 'SystemOverview',
                    'stats': { 'nodes': len(nodes), 'edges': len(edges) },
                    'top_nodes': top
                })
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "search_graph_nodes":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            query = (payload.get('query') or '').strip().lower()
            try:
                limit = int(payload.get('limit') or 10)
            except Exception:
                limit = 10
            gpath = _resolve_system_graph(proj)
            if not gpath:
                return jsonify({"ok": False, "error": "system overview graph not found"}), 404
            try:
                g = json.loads(Path(gpath).read_text())
                nodes = g.get('nodes', []) or g.get('data', {}).get('nodes', [])
                out = []
                for n in nodes:
                    lbl = (n.get('label') or '').lower()
                    typ = (n.get('type') or '').lower()
                    if not query or query in lbl or query in typ:
                        out.append({ 'id': str(n.get('id')), 'label': n.get('label'), 'type': n.get('type') })
                        if len(out) >= limit:
                            break
                return jsonify({ 'ok': True, 'results': out })
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "get_node_details":
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            node_id = str(payload.get('node_id') or '').strip()
            try:
                edge_limit = int(payload.get('edge_limit') or 10)
            except Exception:
                edge_limit = 10
            if not node_id:
                return jsonify({"ok": False, "error": "missing node_id"}), 400
            gpath = _resolve_system_graph(proj)
            if not gpath:
                return jsonify({"ok": False, "error": "system overview graph not found"}), 404
            try:
                g = json.loads(Path(gpath).read_text())
                data = g.get('data') or g
                nodes = { str(n.get('id')): n for n in (data.get('nodes') or []) }
                edges = data.get('edges') or []
                n = nodes.get(node_id)
                if not n:
                    return jsonify({"ok": False, "error": "node not found"}), 404
                # Collect limited edges
                inc = []
                for e in edges:
                    s = str(e.get('source'))
                    t = str(e.get('target'))
                    if s == node_id or t == node_id:
                        inc.append({ 'source': s, 'target': t, 'label': e.get('label') })
                        if len(inc) >= edge_limit:
                            break
                # Light node details
                details = {
                    'id': node_id,
                    'label': n.get('label') or node_id,
                    'type': n.get('type'),
                    'observations': (n.get('observations') or [])[:3],
                    'assumptions': (n.get('assumptions') or [])[:3]
                }
                return jsonify({ 'ok': True, 'node': details, 'edges': inc })
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
        if name == "get_file_snippet":
            # Fetch a small code snippet by relpath or card_id from the cards index
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            rel = (payload.get('relpath') or '').strip()
            cid = (payload.get('card_id') or '').strip()
            try:
                max_bytes = int(payload.get('max_bytes') or 2000)
            except Exception:
                max_bytes = 2000
            id_to_rel, rel_to_id, rel_to_snips = _cards_index(proj)
            if not rel and cid:
                rel = id_to_rel.get(cid, '')
            if not rel:
                return jsonify({"ok": False, "error": "missing relpath/card_id"}), 400
            # Prefer snippet from cards.jsonl if available
            snips = rel_to_snips.get(rel) or []
            text = ''
            if snips:
                text = '\n\n'.join(snips)
            else:
                # Try to read file content relative to project source if present
                try:
                    # Common locations: project has manifest/files.json listing relpaths; files likely under original repo path
                    # Attempt to find a plausible source root: project_dir/.. might contain a copy; otherwise, skip
                    candidate = None
                    # 1) project_dir / source
                    for base in [proj, proj.parent]:
                        p = base / rel
                        if p.exists():
                            candidate = p
                            break
                    if candidate and candidate.exists():
                        buf = candidate.read_text(errors='ignore')
                        text = buf
                except Exception:
                    text = ''
            # Trim
            if len(text.encode('utf-8')) > max_bytes:
                # Keep head and tail
                head = text[: int(max_bytes*0.6)]
                tail = text[- int(max_bytes*0.3):]
                text = head + "\n...\n" + tail
            return jsonify({"ok": True, "relpath": rel, "card_id": rel_to_id.get(rel), "snippet": text})
        if name == "get_recent_activity":
            # Summarize the most recent decision/action from telemetry
            if not proj:
                return jsonify({"ok": False, "error": "no active project"}), 400
            # Find latest instance for this project
            latest = None
            if REGISTRY_DIR.exists():
                for f in sorted(REGISTRY_DIR.glob("*.json")):
                    inst = _read_instance(f)
                    if not inst:
                        continue
                    if inst.get('project_id') == str(proj):
                        if (latest is None) or (float(inst.get('started_at') or 0) > float(latest.get('started_at') or 0)):
                            latest = inst
            if not latest:
                return jsonify({"ok": False, "error": "no running instance for project (start with --telemetry)"}), 400
            tel = latest.get('telemetry', {}) or {}
            ctrl = tel.get('control_url')
            tok = tel.get('token')
            try:
                r = requests.get(f"{ctrl}/recent", params={"limit": 60, **({"token": tok} if tok else {})}, timeout=5)
                ev = r.json().get('events', []) if r.ok else []
            except Exception:
                ev = []
            # Pick the last decision event, else last executing/analyzing/result
            pick = None
            for e in reversed(ev):
                if e.get('type') == 'decision':
                    pick = e
                    break
            if not pick:
                for e in reversed(ev):
                    if e.get('type') in ('executing','analyzing','result','code_loaded'):
                        pick = e
                        break
            if not pick:
                return jsonify({"ok": True, "empty": True, "summary": "No recent activity yet."})
            action = pick.get('action') or pick.get('type')
            reasoning = pick.get('reasoning') or ''
            params = pick.get('parameters') or {}
            # Extract hints
            file_path = params.get('file_path') or ''
            node_ids = params.get('node_ids') or []
            # Friendly label
            def _label(a: str) -> str:
                a = (a or '').lower()
                if a in ('load_nodes','load_node','fetch_code'): return 'Fetch code'
                if a=='update_node': return 'Memorize fact'
                if a in ('add_edge',): return 'Relate facts'
                if a in ('query_graph','focus','summarize'): return 'Explore graph'
                if a=='propose_hypothesis': return 'Hypothesis'
                if a=='update_hypothesis': return 'Hypothesis update'
                if a=='deep_think': return 'Strategist'
                return a or 'Act'
            label = _label(action)
            # Compose text (short)
            bits = [f"Action: {label}"]
            if file_path:
                bits.append(f"File: {file_path}")
            if node_ids:
                bits.append(f"Nodes: {', '.join([str(x) for x in node_ids[:3]])}")
            if reasoning:
                # Trim long thoughts
                if len(reasoning) > 280:
                    reasoning = reasoning[:277] + '...'
                bits.append(f"Thought: {reasoning}")
            summary = "\n".join(bits)
            return jsonify({
                "ok": True,
                "iteration": pick.get('iteration'),
                "action": action,
                "label": label,
                "file_path": file_path or None,
                "node_ids": node_ids[:3] if isinstance(node_ids, list) else [],
                "summary": summary
            })
        return jsonify({"ok": False, "error": f"Unknown tool: {name}"}), 400

    # Dashboard snapshot API
    @app.get("/api/dashboard")
    def dashboard():
        pid = (request.args.get("project") or _read_active_project() or "").strip()
        if not pid:
            return jsonify({"error": "no active project"}), 400
        proj = _project_path(pid)
        if not proj:
            return jsonify({"error": f"project not found: {pid}"}), 404
        sess = _read_latest_session(proj)
        # Plan: take last planning batch
        planning = sess.get('planning_history') or []
        plan_items = []
        if planning:
            last = planning[-1]
            items = last.get('items') or []
            for it in items:
                if isinstance(it, dict) and ('goal' in it or 'description' in it or 'tool_name' in it):
                    goal = it.get('goal') or it.get('description') or it.get('tool_name') or 'Unknown'
                    plan_items.append({
                        'goal': goal,
                        'priority': it.get('priority'),
                        'focus_areas': it.get('focus_areas', []),
                        'reasoning': it.get('reasoning')
                    })
        # Completed investigations
        inv = sess.get('investigations') or []
        completed_goals = set()
        for r in inv:
            g = (r.get('goal') or '').strip()
            if g:
                completed_goals.add(g)
        # Mark done
        for p in plan_items:
            p['done'] = p.get('goal') in completed_goals
        # Current investigation: pick first not-done from plan, fallback to last investigation goal
        current_goal = None
        for p in plan_items:
            if not p.get('done'):
                current_goal = p.get('goal')
                break
        if not current_goal and inv:
            current_goal = inv[-1].get('goal')
        # Top hypotheses from project file
        top_hyps = []
        hyp_file = proj / 'hypotheses.json'
        try:
            if hyp_file.exists():
                data = json.loads(hyp_file.read_text())
                hyps = list((data.get('hypotheses') or {}).values())
                # Sort by confidence desc; exclude rejected
                hyps = [h for h in hyps if h.get('status') != 'rejected']
                hyps.sort(key=lambda h: float(h.get('confidence', 0.0)), reverse=True)
                for h in hyps[:5]:
                    top_hyps.append({
                        'id': h.get('id'),
                        'node_id': h.get('node_id'),
                        'type': h.get('vulnerability_type'),
                        'description': h.get('description'),
                        'confidence': h.get('confidence'),
                        'status': h.get('status')
                    })
        except Exception:
            pass
        # Basic stats from session
        coverage = sess.get('coverage') or {}
        tu = (sess.get('token_usage') or {}).get('total_usage', {})
        out = {
            'project_id': pid,
            'plan': plan_items,
            'current': {'goal': current_goal},
            'top_hypotheses': top_hyps,
            'coverage': coverage,
            'token_usage': {
                'total_tokens': int(tu.get('total_tokens', 0)),
                'call_count': int(tu.get('call_count', 0))
            },
            'session_status': sess.get('status', 'unknown')
        }
        return jsonify(out)

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5280"))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    try:
        print(f"[Hound Chatbot] Starting on http://{host}:{port} (debug={debug})")
        print(f"[Hound Chatbot] OPENAI_API_KEY available: {'yes' if _get_openai_api_key() else 'no'}")
    except Exception:
        pass
    app.run(host=host, port=port, debug=debug, use_reloader=False)
