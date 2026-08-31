# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Canvas report renderer for model-analysis graphs."""

from __future__ import annotations

import json


def render_report(graph: dict) -> str:
    """Render a graph description as one offline HTML document."""
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return _DOCUMENT.replace("__MODEL_GRAPH_DATA__", payload)


_DOCUMENT = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model map · warp-nn</title>
<style>
  :root{color-scheme:light;--ink:#18212f;--muted:#697386;--line:#dbe2ea;--paper:#fbfcfe;--card:rgba(255,255,255,.92);--blue:#3974e8;--shadow:0 18px 55px rgba(25,43,70,.12)}
  *{box-sizing:border-box}html,body{height:100%;margin:0;overflow:hidden}body{font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:#f5f7fa}
  button,input{font:inherit}button{color:inherit}.app{height:100%;display:grid;grid-template-rows:auto 1fr}
  header{z-index:5;display:flex;align-items:center;gap:22px;padding:17px 22px;background:rgba(255,255,255,.9);border-bottom:1px solid var(--line);backdrop-filter:blur(16px)}
  .brand{display:flex;align-items:center;gap:11px;min-width:max-content}.mark{width:31px;height:31px;border-radius:10px;background:linear-gradient(145deg,#5e8ff2,#775ad9);box-shadow:inset 0 0 0 1px rgba(255,255,255,.35),0 5px 13px rgba(72,91,190,.25);position:relative}.mark:after{content:"";position:absolute;inset:8px;border:2px solid white;border-radius:50%}
  .brand b{font-size:14px;letter-spacing:.01em}.brand small{display:block;color:var(--muted);font-size:11px;margin-top:2px}.title{min-width:0;flex:1}.title h1{font-size:16px;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.title p{font-size:12px;color:var(--muted);margin:3px 0 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .summary{display:flex;gap:20px;white-space:nowrap}.metric span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.metric strong{font-size:13px;font-weight:650}
  main{min-height:0;display:grid;grid-template-columns:1fr 360px;position:relative}.stage{position:relative;min-width:0;overflow:hidden;background-color:#f8fafc;background-image:radial-gradient(#d8dee8 1px,transparent 1px);background-size:22px 22px}
  canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}
  .toolbar{position:absolute;z-index:3;left:18px;top:18px;display:flex;gap:8px;align-items:center;padding:7px;background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:13px;box-shadow:0 8px 25px rgba(30,45,70,.09);backdrop-filter:blur(12px)}
  .search{position:relative}.search input{width:220px;border:0;outline:0;padding:8px 30px 8px 10px;background:#f2f5f9;border-radius:8px;font-size:12px;color:var(--ink)}.search kbd{position:absolute;right:8px;top:7px;color:#8a94a5;font:10px ui-monospace,monospace;border:1px solid #d6dce5;background:#fff;border-radius:4px;padding:2px 4px}
  .tool{height:32px;padding:0 10px;border:0;border-radius:8px;background:transparent;font-size:11px;font-weight:600;cursor:pointer}.tool:hover{background:#edf2f8}.tool.active{background:#e7efff;color:#285fc9}.divider{width:1px;height:23px;background:var(--line)}
  .hint{position:absolute;z-index:2;left:19px;bottom:17px;padding:8px 11px;border-radius:10px;background:rgba(255,255,255,.83);border:1px solid rgba(215,222,232,.9);color:var(--muted);font-size:10px;box-shadow:0 5px 18px rgba(30,45,70,.06);pointer-events:none}
  aside{z-index:4;min-width:0;background:var(--card);border-left:1px solid var(--line);overflow:auto;padding:24px 23px 36px;box-shadow:-12px 0 35px rgba(35,50,70,.035)}
  .eyebrow{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--blue);margin-bottom:9px}.detail h2{font-size:21px;line-height:1.22;margin:0 0 7px;letter-spacing:-.018em}.sub{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.explanation{font-family:Georgia,"Times New Roman",serif;font-size:15px;line-height:1.58;color:#344052;margin:21px 0;padding:16px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
  .facts{display:grid;grid-template-columns:1fr 1fr;gap:9px}.fact{padding:11px;background:#f4f7fa;border-radius:10px;min-width:0}.fact span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}.fact b{font-size:12px;overflow-wrap:anywhere}
  .section{margin-top:22px}.section h3{margin:0 0 10px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#596477}.weight-list{display:flex;flex-direction:column;gap:6px}.weight{border:1px solid var(--line);border-radius:9px;padding:9px 10px;background:#fff;cursor:pointer}.weight:hover{border-color:#aebbd0;background:#fbfdff}.weight b{display:block;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.weight span{display:block;color:var(--muted);font-size:10px;margin-top:4px}
  .legend{display:flex;flex-wrap:wrap;gap:8px 13px}.legend-item{font-size:10px;color:var(--muted);display:flex;gap:6px;align-items:center}.swatch{width:9px;height:9px;border-radius:50%}
  .empty{height:100%;display:grid;place-content:center;text-align:center;color:var(--muted)}.empty .orb{width:58px;height:58px;margin:0 auto 15px;border:1px solid #cdd7e5;border-radius:50%;box-shadow:inset 0 0 0 13px #eef3fa}.empty b{color:var(--ink);font-size:14px}.empty p{font-size:11px;max-width:240px;line-height:1.5}
  .notice{margin-top:22px;padding:11px;border-radius:9px;background:#eef5ff;color:#42546f;font-size:10px;line-height:1.45}.notice b{color:#285fc9}
  @media(max-width:900px){main{grid-template-columns:1fr 300px}.summary .metric:nth-child(n+3){display:none}}@media(max-width:680px){main{display:block}.stage{height:58%}aside{height:42%;border-left:0;border-top:1px solid var(--line)}.summary{display:none}.search input{width:145px}.toolbar{right:12px;left:12px;overflow:auto}}
</style>
</head>
<body>
<div class="app">
  <header><div class="brand"><div class="mark"></div><div><b>warp-nn</b><small>Model atlas</small></div></div><div class="title"><h1 id="modelName"></h1><p id="modelPath"></p></div><div class="summary" id="summary"></div></header>
  <main>
    <section class="stage" id="stage">
      <canvas id="canvas" aria-label="Interactive model architecture graph"></canvas>
      <div class="toolbar">
        <div class="search"><input id="search" placeholder="Find a layer or tensor…" aria-label="Search nodes"><kbd>/</kbd></div><div class="divider"></div>
        <button class="tool" id="fit">Fit</button><button class="tool" id="tensors">Weights</button><button class="tool" id="physics">Untangle</button>
      </div>
      <div class="hint">Drag to pan · scroll to zoom · click a node to understand it</div>
    </section>
    <aside id="aside"><div class="empty"><div class="orb"></div><b>Choose a component</b><p>Click any node to see what it does and inspect the learned tensors inside it.</p></div></aside>
  </main>
</div>
<script id="model-data" type="application/json">__MODEL_GRAPH_DATA__</script>
<script>
(() => {
  'use strict';
  const graph=JSON.parse(document.getElementById('model-data').textContent), S=graph.summary;
  const canvas=document.getElementById('canvas'), stage=document.getElementById('stage'), ctx=canvas.getContext('2d');
  const aside=document.getElementById('aside'), search=document.getElementById('search');
  const palette={model:'#6c5dd3',embedding:'#3b76de',input_norm:'#83a5c7',attention:'#2f9d8f',post_attention_norm:'#83a5c7',pre_ffn_norm:'#83a5c7',mlp:'#e28a38',post_ffn_norm:'#83a5c7',normalization:'#83a5c7',convolution:'#c66ab3',final_norm:'#758da9',output:'#d05e75',vision:'#4d9bc1',vae:'#9b72cf',other:'#8793a3',tensor:'#afbac8'};
  const names={model:'Overview',embedding:'Embedding',input_norm:'Normalization',attention:'Attention',post_attention_norm:'Normalization',pre_ffn_norm:'Normalization',mlp:'Feed-forward',post_ffn_norm:'Normalization',normalization:'Normalization',convolution:'Convolution',final_norm:'Final norm',output:'Output',vision:'Vision',vae:'VAE',other:'Other',tensor:'Weight'};
  const fmt=n=>{if(n==null)return '—';for(const [u,s] of [['T',1e12],['B',1e9],['M',1e6],['K',1e3]])if(n>=s)return (n/s).toLocaleString(undefined,{maximumSignificantDigits:4})+u;return Number(n).toLocaleString()};
  const bytes=n=>{if(n==null)return '—';for(const [u,s] of [['TiB',2**40],['GiB',2**30],['MiB',2**20],['KiB',2**10]])if(n>=s)return (n/s).toLocaleString(undefined,{maximumFractionDigits:2})+' '+u;return n+' B'};
  document.title=S.name+' · Model map'; document.getElementById('modelName').textContent=S.name; document.getElementById('modelPath').textContent=S.path;
  const metrics=[['Parameters',fmt(S.parameters)],['Layers',fmt(S.layers)],['Model width',fmt(S.hiddenSize)],['On disk',bytes(S.bytes)]];
  document.getElementById('summary').innerHTML=metrics.map(x=>`<div class="metric"><span>${x[0]}</span><strong>${x[1]}</strong></div>`).join('');
  const nodes=graph.nodes.map(n=>({...n,x:0,y:0,vx:0,vy:0,r:n.type==='tensor'?5:n.kind==='model'?22:11}));
  const byId=new Map(nodes.map(n=>[n.id,n]));
  const componentNodes=nodes.filter(n=>n.type==='component'), tensorNodes=nodes.filter(n=>n.type==='tensor');
  const maxLayer=Math.max(0,...componentNodes.map(n=>n.layer==null?-1:n.layer));
  const laneY=n=>n.kind==='model'?0:(n.layer==null?0:(n.lane-3.5)*76);
  for(const n of componentNodes){
    if(n.kind==='model'){n.x=0;n.y=0}
    else if(n.layer!=null){n.x=430+n.layer*245;n.y=laneY(n)}
    else if(['embedding','vision','vae','other'].includes(n.kind)){n.x=220;n.y=(['embedding','vision','vae','other'].indexOf(n.kind)-.8)*92}
    else {n.x=430+(maxLayer+1)*245+(['final_norm','output'].indexOf(n.kind)+.2)*210;n.y=0}
    n.homeX=n.x;n.homeY=n.y;
  }
  for(const p of componentNodes){
    const children=tensorNodes.filter(n=>n.parent===p.id);
    children.forEach((n,i)=>{const ring=Math.floor(i/12),a=(i%12)/Math.min(12,children.length-ring*12)*Math.PI*2;n.x=p.x+Math.cos(a)*(42+ring*20);n.y=p.y+Math.sin(a)*(42+ring*20);n.homeX=n.x;n.homeY=n.y});
  }
  let dpr=1,width=0,height=0,scale=1,ox=0,oy=0,showWeights=false,selected=null,hovered=null,drag=null,physics=false,animation=0,query='';
  const visible=()=>showWeights?nodes:componentNodes;
  function resize(){const r=stage.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);width=r.width;height=r.height;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.width=width+'px';canvas.style.height=height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
  const screen=n=>({x:n.x*scale+ox,y:n.y*scale+oy});
  const world=(x,y)=>({x:(x-ox)/scale,y:(y-oy)/scale});
  function fit(){const v=visible();if(!v.length)return;let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const n of v){minX=Math.min(minX,n.x-70);maxX=Math.max(maxX,n.x+70);minY=Math.min(minY,n.y-50);maxY=Math.max(maxY,n.y+50)}scale=Math.min(1.15,(width-80)/(maxX-minX),(height-80)/(maxY-minY));ox=width/2-(minX+maxX)/2*scale;oy=height/2-(minY+maxY)/2*scale;draw()}
  function rounded(x,y,w,h,r){ctx.beginPath();ctx.roundRect(x,y,w,h,r)}
  function edge(a,b,contains){const A=screen(a),B=screen(b);ctx.beginPath();ctx.moveTo(A.x,A.y);const dx=(B.x-A.x)*.52;ctx.bezierCurveTo(A.x+dx,A.y,B.x-dx,B.y,B.x,B.y);ctx.strokeStyle=contains?'rgba(150,163,180,.20)':'rgba(92,112,141,.25)';ctx.lineWidth=contains?1:1.4;ctx.setLineDash(contains?[2,4]:[]);ctx.stroke();ctx.setLineDash([])}
  function drawNode(n){const p=screen(n), hit=n===selected||n===hovered, match=query&&((n.fullName||n.label)+' '+(n.subtitle||'')).toLowerCase().includes(query);if(n.type==='tensor'){ctx.beginPath();ctx.arc(p.x,p.y,Math.max(2.5,n.r*scale),0,Math.PI*2);ctx.fillStyle=match?'#e84f76':palette.tensor;ctx.fill();return}
    const w=Math.max(82,Math.min(154,72+n.label.length*3.5))*Math.max(.55,Math.min(1,scale)),h=43*Math.max(.65,Math.min(1,scale));
    if(hit||match){ctx.shadowColor=match?'rgba(232,79,118,.35)':'rgba(39,71,118,.25)';ctx.shadowBlur=18}
    rounded(p.x-w/2,p.y-h/2,w,h,10);ctx.fillStyle='rgba(255,255,255,.97)';ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle=match?'#e84f76':hit?palette[n.kind]:'#d4dce7';ctx.lineWidth=hit||match?2:1;ctx.stroke();
    ctx.beginPath();ctx.arc(p.x-w/2+12,p.y,4.2,0,Math.PI*2);ctx.fillStyle=palette[n.kind]||palette.other;ctx.fill();
    if(scale>.31){ctx.fillStyle='#233044';ctx.font=`600 ${Math.max(8,11*Math.min(1,scale))}px ui-sans-serif,system-ui`;ctx.textAlign='left';ctx.textBaseline='middle';const label=n.label.length>25?n.label.slice(0,24)+'…':n.label;ctx.fillText(label,p.x-w/2+22,p.y-(scale>.6?5:0));if(scale>.6){ctx.fillStyle='#7a8494';ctx.font='9px ui-sans-serif,system-ui';ctx.fillText(n.subtitle,p.x-w/2+22,p.y+9)}}
  }
  function draw(){ctx.clearRect(0,0,width,height);const ids=new Set(visible().map(n=>n.id));for(const e of graph.edges){if(ids.has(e.source)&&ids.has(e.target))edge(byId.get(e.source),byId.get(e.target),e.kind==='contains')}for(const n of visible())drawNode(n)}
  function hitTest(x,y){const v=visible();for(let i=v.length-1;i>=0;i--){const n=v[i],p=screen(n),radius=n.type==='tensor'?Math.max(8,n.r*scale):Math.max(22,55*scale);if((p.x-x)**2+(p.y-y)**2<radius**2)return n}return null}
  function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
  function fact(label,value){return `<div class="fact"><span>${label}</span><b>${esc(value)}</b></div>`}
  function select(n){selected=n;if(!n){aside.innerHTML='<div class="empty"><div class="orb"></div><b>Choose a component</b><p>Click any node to see what it does and inspect the learned tensors inside it.</p></div>';draw();return}
    const children=tensorNodes.filter(x=>x.parent===n.id);let facts='';
    if(n.kind==='model'){facts=fact('Architecture',S.architecture)+fact('Checkpoint',S.format)+fact('Parameters',fmt(S.parameters))+fact('Stored size',bytes(S.bytes))+fact('Transformer layers',S.layers)+fact('Tensor formats',Object.entries(S.formats).map(x=>x.join(' × ')).join(', '))}
    else if(n.type==='tensor'){facts=fact('Shape',(n.shape||[]).join(' × ')||'scalar')+fact('Storage format',n.format)+fact('Values',fmt(n.parameters))+fact('Stored size',bytes(n.bytes))}
    else{facts=fact('Parameters',fmt(n.parameters))+fact('Stored size',bytes(n.bytes))+fact('Tensors',n.tensorCount)+fact('Layer',n.layer==null?'Global':n.layer)+(n.layerType?fact('Layer variant',n.layerType.replaceAll('_',' ')):'')}
    const weights=children.length?`<div class="section"><h3>Learned tensors</h3><div class="weight-list">${children.map(c=>`<div class="weight" data-node="${c.id}"><b>${esc(c.fullName)}</b><span>${esc(c.subtitle)} · ${esc(c.format)} · ${bytes(c.bytes)}</span></div>`).join('')}</div></div>`:'';
    aside.innerHTML=`<div class="detail"><div class="eyebrow">${esc(names[n.kind]||'Component')}</div><h2>${esc(n.type==='tensor'?n.fullName:n.label)}</h2><div class="sub">${esc(n.subtitle)}</div><div class="explanation">${esc(n.explanation)}</div><div class="facts">${facts}</div>${weights}<div class="notice"><b>Reading the map.</b> Solid lines show the main flow of representations. Fine dotted lines connect a component to the weight tables it owns. Parameter sizes come from checkpoint headers; no model tensors were loaded.</div></div>`;
    aside.querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>select(byId.get(el.dataset.node)));draw();
  }
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);const n=hitTest(e.offsetX,e.offsetY);drag={x:e.clientX,y:e.clientY,ox,oy,node:n&&e.altKey?n:null};canvas.classList.add('dragging')});
  canvas.addEventListener('pointermove',e=>{if(drag){if(drag.node){const w=world(e.offsetX,e.offsetY);drag.node.x=w.x;drag.node.y=w.y}else{ox=drag.ox+e.clientX-drag.x;oy=drag.oy+e.clientY-drag.y}draw();return}hovered=hitTest(e.offsetX,e.offsetY);canvas.style.cursor=hovered?'pointer':'grab';draw()});
  canvas.addEventListener('pointerup',e=>{const moved=drag&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4;if(!moved)select(hitTest(e.offsetX,e.offsetY));drag=null;canvas.classList.remove('dragging')});
  canvas.addEventListener('wheel',e=>{e.preventDefault();const before=world(e.offsetX,e.offsetY),factor=Math.exp(-e.deltaY*.001);scale=Math.max(.025,Math.min(3.5,scale*factor));ox=e.offsetX-before.x*scale;oy=e.offsetY-before.y*scale;draw()},{passive:false});
  function physicsStep(){if(!physics)return;const v=componentNodes;for(let i=0;i<v.length;i++){const a=v[i];a.vx+=(a.homeX-a.x)*.002;a.vy+=(a.homeY-a.y)*.002;for(let j=i+1;j<v.length;j++){const b=v[j],dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+1;if(d2<18000){const f=85/d2;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f}}}for(const n of v){n.vx*=.88;n.vy*=.88;n.x+=n.vx;n.y+=n.vy}draw();animation=requestAnimationFrame(physicsStep)}
  document.getElementById('fit').onclick=fit;
  document.getElementById('tensors').onclick=e=>{showWeights=!showWeights;e.currentTarget.classList.toggle('active',showWeights);fit()};
  document.getElementById('physics').onclick=e=>{physics=!physics;e.currentTarget.classList.toggle('active',physics);if(physics){cancelAnimationFrame(animation);physicsStep()}};
  search.addEventListener('input',()=>{query=search.value.trim().toLowerCase();if(query){const found=nodes.find(n=>((n.fullName||n.label)+' '+(n.subtitle||'')).toLowerCase().includes(query));if(found){if(found.type==='tensor'&&!showWeights){showWeights=true;document.getElementById('tensors').classList.add('active')}const p=screen(found);ox+=width/2-p.x;oy+=height/2-p.y;select(found)}}draw()});
  document.addEventListener('keydown',e=>{if(e.key==='/'&&document.activeElement!==search){e.preventDefault();search.focus()}if(e.key==='Escape'){search.value='';query='';search.blur();draw()}});
  const legendKinds=['embedding','attention','mlp','normalization','output','tensor'];
  const legend=document.createElement('div');legend.className='section';legend.innerHTML='<h3>Visual vocabulary</h3><div class="legend">'+legendKinds.map(k=>`<span class="legend-item"><i class="swatch" style="background:${palette[k]}"></i>${names[k]}</span>`).join('')+'</div>';aside.appendChild(legend);
  new ResizeObserver(resize).observe(stage);resize();requestAnimationFrame(()=>{fit();select(byId.get('model'))});
})();
</script>
</body>
</html>
"""
