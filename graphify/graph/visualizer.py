"""
visualizer.py — writes graph.json and a self-contained graph.html.

graph.json  : networkx node_link format (loadable with nx.node_link_graph)
graph.html  : single-file vis-network visualization with dark theme,
              legend, stats panel, search, node inspector, and physics toggle.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

# ── Visual configuration ───────────────────────────────────────────────────────

_NODE_COLOR = {
    "repo":     "#3498DB",
    "file":     "#2ECC71",
    "function": "#F39C12",
    "class":    "#E74C3C",
    "import":   "#9B59B6",
    "section":  "#1ABC9C",
}
_NODE_SHAPE = {
    "repo":     "diamond",
    "file":     "box",
    "function": "dot",
    "class":    "ellipse",
    "import":   "triangle",
    "section":  "star",
}
_NODE_BASE_SIZE = {
    "repo": 28, "file": 18, "function": 11,
    "class": 15, "import": 9, "section": 11,
}
_EDGE_COLOR = {
    "contains":   "#444455",
    "imports":    "#9B59B6",
    "calls":      "#F39C12",
    "inherits":   "#E74C3C",
    "references": "#3498DB",
}


# ── graph.json ────────────────────────────────────────────────────────────────

def write_graph_json(G: nx.MultiDiGraph, output_path: Path) -> None:
    """Serialise graph to networkx node_link JSON."""
    data = json_graph.node_link_data(G, edges="links")
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── graph.html ────────────────────────────────────────────────────────────────

def write_graph_html(G: nx.MultiDiGraph, output_path: Path, title: str = "Graphify") -> None:
    """Write a self-contained vis-network HTML file."""

    # ── Build vis.js node + edge arrays ──────────────────────────────────────
    node_id_map: dict[str, int] = {}
    vis_nodes: list[dict] = []

    for i, (nid, data) in enumerate(G.nodes(data=True)):
        node_id_map[nid] = i
        ntype  = data.get("type", "file")
        name   = data.get("name", nid.split(":")[-1])
        label  = name if len(name) <= 28 else name[:25] + "…"

        # Tooltip (rendered as HTML inside vis)
        tip_rows = [f"<b>{name}</b>", f"Type: {ntype}"]
        if data.get("file_path"):
            tip_rows.append(f"File: {data['file_path']}")
        if data.get("start_line"):
            tip_rows.append(f"Lines: {data['start_line']}–{data.get('end_line','?')}")
        if data.get("language"):
            tip_rows.append(f"Lang: {data['language']}")
        if data.get("metadata"):
            md = data["metadata"]
            if md.get("pipeline_name"):
                tip_rows.append(f"Pipeline: {md['pipeline_name']}")
            if md.get("activity_count"):
                tip_rows.append(f"Activities: {md['activity_count']}")

        # Size = base + sqrt(degree) so hubs pop without overwhelming
        import math
        degree    = G.degree(nid)
        base_size = _NODE_BASE_SIZE.get(ntype, 11)
        size      = min(base_size + math.sqrt(degree) * 3, 55)

        vis_nodes.append({
            "id":     i,
            "label":  label,
            "title":  "<br>".join(tip_rows),
            "color":  _NODE_COLOR.get(ntype, "#888"),
            "shape":  _NODE_SHAPE.get(ntype, "dot"),
            "size":   round(size, 1),
            "group":  ntype,
            # Custom fields for the inspector panel
            "_gid":    nid,
            "_gtype":  ntype,
            "_gname":  name,
            "_gfile":  data.get("file_path", ""),
            "_grepo":  data.get("repo", ""),
            "_glang":  data.get("language", ""),
            "_gstart": data.get("start_line", 0),
            "_gend":   data.get("end_line", 0),
            "_gmeta":  data.get("metadata", {}),
        })

    vis_edges: list[dict] = []
    for ei, (u, v, data) in enumerate(G.edges(data=True)):
        if u not in node_id_map or v not in node_id_map:
            continue
        etype = data.get("type", "references")
        vis_edges.append({
            "id":     ei,
            "from":   node_id_map[u],
            "to":     node_id_map[v],
            "title":  etype,
            "color":  {"color": _EDGE_COLOR.get(etype, "#555"), "opacity": 0.55},
            "arrows": "to",
            "width":  1.8 if etype == "calls" else 1.0,
            "dashes": etype in ("imports", "references"),
        })

    # Stats for the sidebar
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "?")
        node_types[t] = node_types.get(t, 0) + 1
    for _, _, d in G.edges(data=True):
        t = d.get("type", "?")
        edge_types[t] = edge_types.get(t, 0) + 1

    nodes_json = json.dumps(vis_nodes, default=str)
    edges_json = json.dumps(vis_edges, default=str)
    stats_json = json.dumps({
        "nodes":      G.number_of_nodes(),
        "edges":      G.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
    })

    # ── HTML template ──────────────────────────────────────────────────────────
    # Note: {{ and }} are f-string escapes for literal { and } in the JS/CSS.
    # ${{}}<expr>{{}} becomes ${<expr>} in the output (JS template literal).

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Graphify</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;overflow:hidden;display:flex;flex-direction:column}}
#hdr{{background:#161b22;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #30363d;flex-shrink:0}}
#hdr h1{{font-size:15px;color:#58a6ff;font-weight:600;letter-spacing:.3px}}
#hdr .sub{{font-size:11px;color:#8b949e}}
#main{{flex:1;display:flex;overflow:hidden}}
#sidebar{{width:272px;background:#161b22;border-right:1px solid #30363d;overflow-y:auto;display:flex;flex-direction:column;flex-shrink:0}}
#gc{{flex:1}}
.sec{{padding:11px 12px;border-bottom:1px solid #21262d}}
.sec h3{{font-size:9.5px;text-transform:uppercase;letter-spacing:.8px;color:#8b949e;margin-bottom:8px}}
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.si{{background:#0d1117;border-radius:5px;padding:7px;text-align:center}}
.sv{{font-size:18px;font-weight:700;color:#58a6ff}}
.sl{{font-size:9.5px;color:#8b949e;margin-top:1px}}
.leg-item{{display:flex;align-items:center;gap:7px;padding:3px 4px;font-size:11.5px;cursor:pointer;border-radius:4px}}
.leg-item:hover{{background:#21262d}}
.leg-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
#sw{{padding:8px 12px}}
#si{{width:100%;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 10px;border-radius:6px;font-size:12px}}
#si:focus{{outline:none;border-color:#58a6ff}}
#sr{{padding:0 12px 6px}}
.sr-item{{padding:4px 7px;border-radius:4px;font-size:11.5px;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.sr-item:hover{{background:#21262d}}
#nd{{padding:12px;flex:1}}
#nd h3{{font-size:9.5px;text-transform:uppercase;letter-spacing:.8px;color:#8b949e;margin-bottom:8px}}
#ni{{font-size:11.5px;line-height:1.9}}
.ir{{display:flex;gap:8px}}
.ik{{color:#8b949e;min-width:55px;flex-shrink:0}}
.iv{{color:#e6edf3;word-break:break-all}}
#tb{{display:flex;gap:5px;padding:8px 12px;border-top:1px solid #21262d;flex-shrink:0}}
.btn{{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:5px 8px;border-radius:5px;cursor:pointer;font-size:10.5px;flex:1;text-align:center;user-select:none}}
.btn:hover{{background:#30363d}}
</style>
</head>
<body>
<div id="hdr">
  <h1>⬡ Graphify</h1>
  <span class="sub">{title}</span>
  <span class="sub" style="margin-left:auto">Generated {generated}</span>
</div>
<div id="main">
  <div id="sidebar">
    <div id="sw"><input id="si" type="text" placeholder="Search nodes…"></div>
    <div id="sr"></div>
    <div class="sec">
      <h3>Statistics</h3>
      <div class="stat-grid" id="sg"></div>
    </div>
    <div class="sec">
      <h3>Node Types  <span style="color:#444;font-weight:300">(click to filter)</span></h3>
      <div id="leg"></div>
    </div>
    <div id="nd">
      <h3>Selected Node</h3>
      <div id="ni"><span style="color:#8b949e;font-size:11px">Click a node to inspect it</span></div>
    </div>
    <div id="tb">
      <div class="btn" onclick="network.fit()">Fit</div>
      <div class="btn" onclick="togglePhysics()">Physics</div>
      <div class="btn" onclick="resetHL()">Reset</div>
    </div>
  </div>
  <div id="gc"></div>
</div>
<script>
const GN = {nodes_json};
const GE = {edges_json};
const GS = {stats_json};

const NC = {{repo:'#3498DB',file:'#2ECC71','function':'#F39C12','class':'#E74C3C',import:'#9B59B6',section:'#1ABC9C'}};

const nDS = new vis.DataSet(GN);
const eDS = new vis.DataSet(GE);
const net = new vis.Network(
  document.getElementById('gc'),
  {{nodes:nDS,edges:eDS}},
  {{
    nodes:{{font:{{size:11,color:'#c9d1d9'}},borderWidth:1.5,borderWidthSelected:3}},
    edges:{{smooth:{{type:'continuous',roundness:0.2}},font:{{size:9,color:'#8b949e',align:'middle'}},selectionWidth:2}},
    physics:{{
      enabled:true,
      solver:'forceAtlas2Based',
      forceAtlas2Based:{{gravitationalConstant:-32,centralGravity:0.003,springLength:160,springConstant:0.07,avoidOverlap:0.6}},
      stabilization:{{iterations:200,updateInterval:50}},
      maxVelocity:70,minVelocity:0.5
    }},
    interaction:{{hover:true,tooltipDelay:120,hideEdgesOnDrag:true,multiselect:true}}
  }}
);

net.on('stabilizationIterationsDone',()=>net.setOptions({{physics:{{enabled:false}}}}));

// ── Stats ──
document.getElementById('sg').innerHTML=
  `<div class="si"><div class="sv">${{GS.nodes}}</div><div class="sl">Nodes</div></div>`+
  `<div class="si"><div class="sv">${{GS.edges}}</div><div class="sl">Edges</div></div>`+
  Object.entries(GS.node_types).map(([k,v])=>
    `<div class="si"><div class="sv">${{v}}</div><div class="sl">${{k}}s</div></div>`
  ).join('');

// ── Legend ──
const legEl = document.getElementById('leg');
Object.entries(NC).forEach(([type,color])=>{{
  const d=document.createElement('div');
  d.className='leg-item';
  d.innerHTML=`<div class="leg-dot" style="background:${{color}}"></div>${{type}}`;
  d.onclick=()=>filterByType(type);
  legEl.appendChild(d);
}});

// ── Node click ──
const niEl = document.getElementById('ni');
net.on('selectNode',({{nodes:sel}})=>{{
  if(!sel.length)return;
  const n=nDS.get(sel[0]);
  const nb=net.getConnectedNodes(sel[0]);
  nDS.update(nDS.get().map(x=>({{id:x.id,opacity:nb.includes(x.id)||x.id===sel[0]?1:0.12}})));

  let meta='';
  if(n._gmeta && n._gmeta.pipeline_name)
    meta+=`<div class="ir"><span class="ik">Pipeline</span><span class="iv">${{n._gmeta.pipeline_name}}</span></div>`;
  if(n._gmeta && n._gmeta.activity_count)
    meta+=`<div class="ir"><span class="ik">Activities</span><span class="iv">${{n._gmeta.activity_count}}</span></div>`;
  if(n._gmeta && n._gmeta.activity_names && n._gmeta.activity_names.length)
    meta+=`<div class="ir"><span class="ik">Acts</span><span class="iv" style="font-size:10px">${{n._gmeta.activity_names.slice(0,5).join(', ')}}</span></div>`;

  niEl.innerHTML=
    `<div class="ir"><span class="ik">Name</span><span class="iv">${{n._gname||n.label}}</span></div>`+
    `<div class="ir"><span class="ik">Type</span><span class="iv" style="color:${{NC[n._gtype]||'#ccc'}}">${{n._gtype}}</span></div>`+
    (n._grepo?`<div class="ir"><span class="ik">Repo</span><span class="iv">${{n._grepo}}</span></div>`:'')+
    (n._gfile?`<div class="ir"><span class="ik">File</span><span class="iv">${{n._gfile}}</span></div>`:'')+
    (n._glang?`<div class="ir"><span class="ik">Lang</span><span class="iv">${{n._glang}}</span></div>`:'')+
    (n._gstart?`<div class="ir"><span class="ik">Lines</span><span class="iv">${{n._gstart}}–${{n._gend}}</span></div>`:'')+
    meta+
    `<div class="ir" style="margin-top:6px"><span class="ik">Degree</span><span class="iv">${{nb.length}}</span></div>`;
}});
net.on('deselectNode',resetHL);

function resetHL(){{
  nDS.update(nDS.get().map(n=>({{id:n.id,opacity:1}})));
  niEl.innerHTML='<span style="color:#8b949e;font-size:11px">Click a node to inspect it</span>';
}}

// ── Search ──
const siEl=document.getElementById('si');
const srEl=document.getElementById('sr');
siEl.addEventListener('input',()=>{{
  const q=siEl.value.toLowerCase().trim();
  srEl.innerHTML='';
  if(!q)return;
  nDS.get().filter(n=>
    (n._gname||'').toLowerCase().includes(q)||
    (n._gfile||'').toLowerCase().includes(q)||
    (n._gtype||'').toLowerCase().includes(q)
  ).slice(0,12).forEach(n=>{{
    const el=document.createElement('div');
    el.className='sr-item';
    el.innerHTML=`<span style="color:${{NC[n._gtype]||'#aaa'}};font-size:10px">${{n._gtype}}</span> ${{n._gname||n.label}}`;
    el.onclick=()=>{{net.selectNodes([n.id]);net.focus(n.id,{{scale:1.4,animation:true}});srEl.innerHTML='';siEl.value=''}};
    srEl.appendChild(el);
  }});
}});

// ── Filter by type ──
function filterByType(type){{
  nDS.update(nDS.get().map(n=>({{id:n.id,opacity:n._gtype===type?1:0.1}})));
}}

// ── Physics toggle ──
let phys=false;
function togglePhysics(){{phys=!phys;net.setOptions({{physics:{{enabled:phys}}}})}}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ── GRAPH_REPORT.md ────────────────────────────────────────────────────────────

def write_graph_report(
    G: nx.MultiDiGraph,
    output_path: Path,
    repos: list[str],
) -> None:
    """Write GRAPH_REPORT.md with stats, top nodes, and suggested questions."""
    from graphify.graph.builder import (
        graph_stats,
        isolated_nodes,
        most_imported,
        top_nodes_by_degree,
    )

    stats    = graph_stats(G)
    top      = top_nodes_by_degree(G, 12)
    imports  = most_imported(G, 10)
    isolated = isolated_nodes(G)

    lines = [
        "# Graph Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Repos:** {', '.join(repos)}",
        "",
        "## Statistics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total nodes | {stats['total_nodes']} |",
        f"| Total edges | {stats['total_edges']} |",
    ]
    for t, c in sorted(stats["node_types"].items()):
        lines.append(f"| {t.capitalize()}s | {c} |")
    for t, c in sorted(stats["edge_types"].items()):
        lines.append(f"| {t.capitalize()} edges | {c} |")

    lines += [
        "",
        "## Most Connected Nodes",
        "",
        "| # | Name | Type | Degree |",
        "|---|------|------|-------:|",
    ]
    for i, (nid, deg, data) in enumerate(top, 1):
        name  = data.get("name", nid.split(":")[-1])
        ntype = data.get("type", "?")
        lines.append(f"| {i} | `{name}` | {ntype} | {deg} |")

    if imports:
        lines += [
            "",
            "## Most Imported Modules",
            "",
            "| # | Module | Imported by |",
            "|---|--------|------------:|",
        ]
        for i, (nid, deg, data) in enumerate(imports, 1):
            name = data.get("name", nid.split(":")[-1])
            lines.append(f"| {i} | `{name}` | {deg} |")

    if isolated:
        lines += [
            "",
            f"## Isolated Nodes ({len(isolated)})",
            "",
            "These nodes have no connections — possible dead code or standalone scripts:",
            "",
        ]
        for nid, data in isolated[:15]:
            name  = data.get("name", nid.split(":")[-1])
            ntype = data.get("type", "?")
            fpath = data.get("file_path", "")
            lines.append(f"- `{name}` ({ntype}){' — ' + fpath if fpath else ''}")

    # Suggested questions derived from graph structure
    lines += ["", "## Suggested Questions", ""]

    file_names = [
        data.get("name", "")
        for _, data in G.nodes(data=True)
        if data.get("type") == "file" and data.get("name")
    ]
    func_names = [
        data.get("name", "")
        for _, data in G.nodes(data=True)
        if data.get("type") == "function" and data.get("name")
    ]
    import_names = [
        data.get("name", "")
        for _, data in G.nodes(data=True)
        if data.get("type") == "import" and data.get("name")
    ]

    questions = []
    if file_names:
        questions.append(f'- "What does `{file_names[0]}` do?"')
    if func_names:
        questions.append(f'- "Where is `{func_names[0]}` called from?"')
    if import_names:
        questions.append(f'- "Which files import `{import_names[0]}`?"')
    if len(file_names) >= 2:
        questions.append(f'- "How do `{file_names[0]}` and `{file_names[1]}` interact?"')
    if not questions:
        questions.append('- "What are the main entry points in this codebase?"')

    lines.extend(questions)

    output_path.write_text("\n".join(lines), encoding="utf-8")
