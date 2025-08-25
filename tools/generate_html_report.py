#!/usr/bin/env python3
"""
Generate a rich, self-contained HTML report from a Hound score JSON file.

Usage:
  python tools/generate_html_report.py --input /path/to/score.json --output report.html --title "Project Report"

The generated HTML embeds all assets (CSS/JS) and requires no network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from html import escape
from string import Template


def load_score(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(num: float, den: float) -> float:
    if not den:
        return 0.0
    return (num / den) * 100.0


def render_html(data: dict, title: str, source_path: str | None) -> str:
    total_expected = data.get("total_expected", 0)
    total_found = data.get("total_found", 0)
    tp = data.get("true_positives", 0)
    fn = data.get("false_negatives", 0)
    fp = data.get("false_positives", 0)
    detection_rate = data.get("detection_rate", 0.0)
    summary_text = data.get("summary", "")

    matched = data.get("matched_findings", []) or []
    missed = data.get("missed_findings", []) or []
    extra = data.get("extra_findings", []) or []

    # Confidence distribution buckets (0-100 by 10s)
    conf_buckets = [0] * 11
    for m in matched:
        c = m.get("confidence", 0)
        try:
            c = float(c)
        except Exception:
            c = 0.0
        idx = int(round(max(0.0, min(1.0, c)) * 10))
        conf_buckets[idx] += 1

    # Severity distribution (from missed_findings if present)
    sev_order = ["critical", "high", "medium", "low", "unknown"]
    sev_counts = {s: 0 for s in sev_order}
    for item in missed:
        sev = str(item.get("severity", "unknown")).lower()
        if sev not in sev_counts:
            sev = "unknown"
        sev_counts[sev] += 1

    # Build list sections
    def render_matched_item(i, it):
        exp_title = it.get("expected", "")
        exp_desc = it.get("expected_description", "")
        match_title = it.get("matched", "")
        match_desc = it.get("matched_description", "")
        just = it.get("justification", "")
        conf = it.get("confidence", 0)
        try:
            conf_pct = int(round(float(conf) * 100))
        except Exception:
            conf_pct = 0
        # Additional fields (ensure ALL details are shown)
        known = {"expected","expected_description","matched","matched_description","justification","confidence"}
        extras = {k:v for k,v in it.items() if k not in known}
        parts = [
            "<details class=card open>",
            "<summary>",
            "<div class=row>",
            "<div class=col>",
            "<div class=eyebrow>Matched Finding #" + str(i + 1) + "</div>",
            "<div class=title>" + escape(exp_title)[:300] + "</div>",
            "</div>",
            "<div class=col-auto>",
            "<div class=badge badge-success>Confidence " + str(conf_pct) + "%</div>",
            "</div>",
            "</div>",
            "</summary>",
            "<div class=content>",
            "<div class=subtle>Expected</div>",
            "<pre class=blk>" + escape(exp_desc) + "</pre>",
            "<div class=subtle>Matched</div>",
            "<pre class=blk>" + escape(match_desc) + "</pre>",
            "<div class=subtle>Justification</div>",
            "<pre class=blk>" + escape(just) + "</pre>",
        ]
        if extras:
            parts.append("<div class=subtle>Additional Fields</div><pre class=blk>" + escape(json.dumps(extras, ensure_ascii=False, indent=2)) + "</pre>")
        parts.extend(["</div>", "</details>"])
        return "".join(parts)

    def render_missed_item(i, it):
        title_ = it.get("title", "")
        sev = str(it.get("severity", "unknown")).lower()
        reason = it.get("reason", "")
        sev_class = {
            "critical": "badge-crit",
            "high": "badge-high",
            "medium": "badge-med",
            "low": "badge-low",
        }.get(sev, "badge-muted")
        known = {"title","severity","reason"}
        extras = {k:v for k,v in it.items() if k not in known}
        parts = [
            "<details class=card>",
            "<summary>",
            "<div class=row>",
            "<div class=col>",
            "<div class=eyebrow>Missed Finding #" + str(i + 1) + "</div>",
            "<div class=title>" + escape(title_) + "</div>",
            "</div>",
            "<div class=col-auto>",
            "<div class=\"badge " + sev_class + "\">" + escape(sev.title()) + "</div>",
            "</div>",
            "</div>",
            "</summary>",
            "<div class=content>",
            "<div class=subtle>Reason</div>",
            "<pre class=blk>" + escape(reason) + "</pre>",
        ]
        if extras:
            parts.append("<div class=subtle>Additional Fields</div><pre class=blk>" + escape(json.dumps(extras, ensure_ascii=False, indent=2)) + "</pre>")
        parts.extend(["</div>", "</details>"])
        return "".join(parts)

    def render_extra_item(i, it):
        fid = it.get("id", "")
        title_ = it.get("title", "")
        assess = str(it.get("assessment", "")).strip()
        known = {"id","title","assessment"}
        extras = {k:v for k,v in it.items() if k not in known}
        parts = [
            "<details class=card>",
            "<summary>",
            "<div class=row>",
            "<div class=col>",
            "<div class=eyebrow>Extra Finding #" + str(i + 1) + "</div>",
            "<div class=title>" + escape(title_) + "</div>",
            "</div>",
            "<div class=col-auto>",
            "<div class=badge>" + escape(assess or 'extra') + "</div>",
            "</div>",
            "</div>",
            "</summary>",
            "<div class=content>",
            "<div class=subtle>Finding ID</div>",
            "<code class=code>" + escape(fid) + "</code>",
        ]
        if extras:
            parts.append("<div class=subtle>Additional Fields</div><pre class=blk>" + escape(json.dumps(extras, ensure_ascii=False, indent=2)) + "</pre>")
        parts.extend(["</div>", "</details>"])
        return "".join(parts)

    matched_html = "\n".join(render_matched_item(i, it) for i, it in enumerate(matched)) or "<div class=muted>No matches.</div>"
    missed_html = "\n".join(render_missed_item(i, it) for i, it in enumerate(missed)) or "<div class=muted>No missed findings.</div>"
    extra_html = "\n".join(render_extra_item(i, it) for i, it in enumerate(extra)) or "<div class=muted>No extra findings.</div>"

    # Compute donut segments (server-side; set as inline styles)
    total_all = max(1, tp + fp + fn)
    tp_deg = (tp / total_all) * 360
    fp_deg = (fp / total_all) * 360
    fn_deg = (fn / total_all) * 360
    tp_end = tp_deg
    fp_end = tp_deg + fp_deg
    fn_end = tp_deg + fp_deg + fn_deg
    donut_main_bg = (
        "conic-gradient(" 
        + f"var(--good) 0 {tp_end:.2f}deg, "
        + f"var(--warn) {tp_end:.2f}deg {fp_end:.2f}deg, "
        + f"var(--bad) {fp_end:.2f}deg {fn_end:.2f}deg, #1b2237 0)"
    )

    # Severity donut for missed
    s_total = sum(sev_counts.values()) or 1
    crit_deg = (sev_counts["critical"] / s_total) * 360
    high_deg = (sev_counts["high"] / s_total) * 360
    med_deg  = (sev_counts["medium"] / s_total) * 360
    low_deg  = (sev_counts["low"] / s_total) * 360
    sev_cum1 = crit_deg
    sev_cum2 = crit_deg + high_deg
    sev_cum3 = crit_deg + high_deg + med_deg
    sev_cum4 = crit_deg + high_deg + med_deg + low_deg
    donut_sev_bg = (
        "conic-gradient(" 
        + f"var(--crit) 0 {sev_cum1:.2f}deg, "
        + f"var(--high) {sev_cum1:.2f}deg {sev_cum2:.2f}deg, "
        + f"var(--med) {sev_cum2:.2f}deg {sev_cum3:.2f}deg, "
        + f"var(--low) {sev_cum3:.2f}deg {sev_cum4:.2f}deg, #1b2237 0)"
    )

    # Precompute percent widths for confidence bars
    conf_total = max(1, sum(conf_buckets))
    conf_widths = [pct(conf_buckets[i], conf_total) for i in range(len(conf_buckets))]

    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    source_block = f"<div class=source>Source: <code>{escape(source_path)}</code></div>" if source_path else ""
    summary_block = (
        f"<section class=panel style=\"margin-top:12px\"><h2>Summary</h2><div class=muted>{escape(summary_text)}</div></section>"
        if summary_text else ""
    )

    tmpl = Template(
        """<!doctype html>
<html lang=en>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1">
  <title>$title</title>
  <style>
    :root {
      --bg: #0b0d12;
      --panel: #121621;
      --panel-2: #0f1320;
      --text: #e6ebf3;
      --muted: #a0a8b8;
      --border: #1f2636;
      --accent: #6ea8fe;
      --good: #32d583;
      --warn: #f59e0b;
      --bad: #ef4444;
      --crit: #ff4d4f;
      --high: #ff7a45;
      --med: #fbbf24;
      --low: #22c55e;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Inter,system-ui,'Helvetica Neue',Arial; }
    a { color: var(--accent); text-decoration: none; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }
    header { display:flex; justify-content: space-between; align-items:center; margin-bottom: 18px; gap: 12px; }
    .titlebar { display:flex; align-items:center; gap: 12px; flex-wrap: wrap; }
    .titlebar h1 { font-size: 22px; margin: 0; font-weight: 650; }
    .subtitle { color: var(--muted); font-size: 12px; }
    .source { color: var(--muted); font-size: 12px; }

    .grid { display: grid; grid-template-columns: repeat(12,1fr); gap: 14px; }
    .col-12 { grid-column: span 12; }
    .col-6 { grid-column: span 6; }
    .col-4 { grid-column: span 4; }
    @media (max-width: 900px) {
      .col-6,.col-4 { grid-column: span 12; }
    }

    .panel { background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
    .panel h2 { margin: 0 0 10px; font-size: 16px; }
    .row { display:flex; align-items:center; justify-content: space-between; gap: 12px; }
    .col { flex:1; min-width: 0; }
    .col-auto { flex: none; }
    .muted { color: var(--muted); }
    .eyebrow { font-size: 11px; color: var(--muted); letter-spacing: .03em; text-transform: uppercase; }
    .title { font-weight: 600; font-size: 14px; }
    .subtle { color: var(--muted); margin: 6px 0 6px; font-size: 12px; }
    .badge { display:inline-block; padding: 4px 8px; border-radius: 999px; background: #1c2335; color: #c3d3ff; font-size: 11px; border: 1px solid var(--border); }
    .badge-success { background: rgba(50,213,131,.12); color: #9ff5cb; border-color: rgba(50,213,131,.25); }
    .badge-muted { background: #1a2234; color: var(--muted); }
    .badge-crit { background: rgba(255,77,79,.15); color: #ffb3b3; border-color: rgba(255,77,79,.3);} 
    .badge-high { background: rgba(255,122,69,.15); color: #ffc6b0; border-color: rgba(255,122,69,.3);} 
    .badge-med { background: rgba(251,191,36,.15); color: #ffe6a6; border-color: rgba(251,191,36,.3);} 
    .badge-low { background: rgba(34,197,94,.15); color: #b7f5cd; border-color: rgba(34,197,94,.3);} 

    .cards { display:grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin: 8px 0 4px; }
    @media (max-width: 1100px) { .cards { grid-template-columns: repeat(2,1fr); } }
    @media (max-width: 680px) { .cards { grid-template-columns: 1fr; } }
    .card-stat { background: #0e1320; border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
    .card-stat .k { font-size: 22px; font-weight: 700; }
    .card-stat .lbl { font-size: 12px; color: var(--muted); }
    .progress { height: 8px; background: #121a2d; border-radius: 999px; overflow:hidden; border:1px solid var(--border); }
    .progress > span { display:block; height:100%; background: linear-gradient(90deg, #7aa8ff, #a58bff); width: 0; }

    .donut { width: 140px; height: 140px; border-radius:50%; position: relative; margin: 6px auto; background: #1b2237; }
    .donut::after { content: ""; position:absolute; inset: 16px; background: var(--panel); border-radius:50%; box-shadow: inset 0 0 0 1px var(--border); }
    .legend { display:flex; gap: 10px; flex-wrap: wrap; justify-content: center; font-size: 12px; color: var(--muted); }
    .lg { display:flex; align-items:center; gap:6px; }
    .swatch { width:10px; height:10px; border-radius:2px; display:inline-block; }

    details.card { background:#0c1220; border:1px solid var(--border); border-radius:12px; padding: 10px 12px; margin: 10px 0; }
    details.card > summary { list-style: none; cursor: pointer; }
    details.card > summary::-webkit-details-marker { display:none; }
    details.card .content { padding-top: 10px; }
    .blk { white-space: pre-wrap; background:#0b1222; border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; }
    .code { background:#0b1222; padding: 2px 6px; border-radius:6px; border:1px solid var(--border); }

    .barlist { display:grid; grid-template-columns: 90px 1fr 40px; gap:10px; align-items:center; }
    .bar { height: 10px; background:#10182a; border:1px solid var(--border); border-radius:999px; overflow:hidden; }
    .bar > span { display:block; height:100%; background: linear-gradient(90deg, #7aa8ff, #a58bff); width:0; }
    .sep { height: 1px; background: var(--border); margin: 14px 0; }
    footer { margin-top: 24px; color: var(--muted); font-size: 12px; text-align:center; }
  </style>
</head>
<body>
  <div class=wrap>
    <header>
      <div class=titlebar>
        <h1>$title</h1>
        <div class=subtitle>Generated $gen_time</div>
      </div>
      $source_block
    </header>

    <section class=panel>
      <div class=cards>
        <div class=card-stat>
          <div class=k>$total_expected</div>
          <div class=lbl>Total Expected</div>
        </div>
        <div class=card-stat>
          <div class=k>$total_found</div>
          <div class=lbl>Total Found</div>
        </div>
        <div class=card-stat>
          <div class=k>$tp</div>
          <div class=lbl>True Positives</div>
        </div>
        <div class=card-stat>
          <div class=k>$fp</div>
          <div class=lbl>False Positives</div>
        </div>
        <div class=card-stat>
          <div class=k>$fn</div>
          <div class=lbl>False Negatives</div>
        </div>
      </div>
      <div style="margin-top:10px">
        <div class=subtle>Detection Rate</div>
        <div class=progress><span id=dr style="width:$detection_rate_pct"></span></div>
      </div>
    </section>

    <div class=grid style="margin-top:12px">
      <div class="col-6 panel">
        <h2>Outcome Mix</h2>
        <div class=donut id=donut-main style="background: $donut_main_bg"></div>
        <div class=legend>
          <div class=lg><span class=swatch style="background:var(--good)"></span> True Positives ($tp)</div>
          <div class=lg><span class=swatch style="background:var(--warn)"></span> False Positives ($fp)</div>
          <div class=lg><span class=swatch style="background:var(--bad)"></span> False Negatives ($fn)</div>
        </div>
      </div>
      <div class="col-6 panel">
        <h2>Missed Severity</h2>
        <div class=donut id=donut-sev style="background: $donut_sev_bg"></div>
        <div class=legend>
          <div class=lg><span class=swatch style="background:var(--crit)"></span> Critical ($sev_critical)</div>
          <div class=lg><span class=swatch style="background:var(--high)"></span> High ($sev_high)</div>
          <div class=lg><span class=swatch style="background:var(--med)"></span> Medium ($sev_medium)</div>
          <div class=lg><span class=swatch style="background:var(--low)"></span> Low ($sev_low)</div>
          <div class=lg><span class=swatch style="background:#1b2237"></span> Unknown ($sev_unknown)</div>
        </div>
      </div>

      <div class="col-12 panel" style="margin-top: 8px;">
        <h2>Confidence Distribution (Matched)</h2>
        <div class=barlist>
          <div class=muted>0–10%</div><div class=bar><span style="width:$w1%"></span></div><div class=muted>$b1</div>
          <div class=muted>10–20%</div><div class=bar><span style="width:$w2%"></span></div><div class=muted>$b2</div>
          <div class=muted>20–30%</div><div class=bar><span style="width:$w3%"></span></div><div class=muted>$b3</div>
          <div class=muted>30–40%</div><div class=bar><span style="width:$w4%"></span></div><div class=muted>$b4</div>
          <div class=muted>40–50%</div><div class=bar><span style="width:$w5%"></span></div><div class=muted>$b5</div>
          <div class=muted>50–60%</div><div class=bar><span style="width:$w6%"></span></div><div class=muted>$b6</div>
          <div class=muted>60–70%</div><div class=bar><span style="width:$w7%"></span></div><div class=muted>$b7</div>
          <div class=muted>70–80%</div><div class=bar><span style="width:$w8%"></span></div><div class=muted>$b8</div>
          <div class=muted>80–90%</div><div class=bar><span style="width:$w9%"></span></div><div class=muted>$b9</div>
          <div class=muted>90–100%</div><div class=bar><span style="width:$w10%"></span></div><div class=muted>$b10</div>
        </div>
      </div>
    </div>

    $summary_block

    <section class=panel style="margin-top:12px">
      <h2>Matched Findings</h2>
      $matched_html
    </section>

    <section class=panel style="margin-top:12px">
      <h2>Missed Findings</h2>
      $missed_html
    </section>

    <section class=panel style="margin-top:12px">
      <h2>Extra Findings</h2>
      $extra_html
    </section>

    <footer>
      <div>Hound Report • $gen_time</div>
    </footer>
  </div>
</body>
</html>
"""
    )

    html = tmpl.substitute(
        title=escape(title),
        gen_time=escape(gen_time),
        source_block=source_block,
        total_expected=total_expected,
        total_found=total_found,
        tp=tp,
        fp=fp,
        fn=fn,
        detection_rate_pct=f"{detection_rate*100:.1f}%",
        donut_main_bg=donut_main_bg,
        sev_critical=sev_counts["critical"],
        sev_high=sev_counts["high"],
        sev_medium=sev_counts["medium"],
        sev_low=sev_counts["low"],
        sev_unknown=sev_counts["unknown"],
        donut_sev_bg=donut_sev_bg,
        w1=f"{conf_widths[1]:.1f}", b1=conf_buckets[1],
        w2=f"{conf_widths[2]:.1f}", b2=conf_buckets[2],
        w3=f"{conf_widths[3]:.1f}", b3=conf_buckets[3],
        w4=f"{conf_widths[4]:.1f}", b4=conf_buckets[4],
        w5=f"{conf_widths[5]:.1f}", b5=conf_buckets[5],
        w6=f"{conf_widths[6]:.1f}", b6=conf_buckets[6],
        w7=f"{conf_widths[7]:.1f}", b7=conf_buckets[7],
        w8=f"{conf_widths[8]:.1f}", b8=conf_buckets[8],
        w9=f"{conf_widths[9]:.1f}", b9=conf_buckets[9],
        w10=f"{conf_widths[10]:.1f}", b10=conf_buckets[10],
        summary_block=summary_block,
        matched_html=matched_html,
        missed_html=missed_html,
        extra_html=extra_html,
    )

    return html


def main():
    ap = argparse.ArgumentParser(description="Generate a beautiful HTML report from a score JSON file.")
    ap.add_argument("--input", required=True, help="Path to score_*.json produced by score_calculator.py")
    ap.add_argument("--output", help="Output HTML path (default: alongside input with .html)")
    ap.add_argument("--title", help="Report title (default: derived from filename)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    data = load_score(in_path)
    title = args.title or f"Hound Report • {in_path.stem}"
    html = render_html(data, title=title, source_path=str(in_path))

    out_path = Path(args.output) if args.output else in_path.with_suffix(".html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
