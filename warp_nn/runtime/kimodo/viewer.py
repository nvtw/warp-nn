# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Write portable, browser-based previews for generated Kimodo motions."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .runner import _SOMA30_PARENTS


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kimodo motion</title>
<style>
:root{color-scheme:light;--ink:#172033;--muted:#697386;--line:#dfe4ee;--panel:rgba(255,255,255,.82);--violet:#6157eb;--cyan:#08a8c7}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden}body{font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f6f7fb;color:var(--ink)}
#stage{position:fixed;inset:0}canvas{display:block;width:100%;height:100%}
.top{position:fixed;z-index:2;left:24px;right:24px;top:20px;display:flex;align-items:flex-start;justify-content:space-between;pointer-events:none}
.brand,.facts,.transport{border:1px solid rgba(203,211,225,.72);background:var(--panel);box-shadow:0 14px 45px rgba(42,51,78,.10);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px)}
.brand{max-width:min(620px,calc(100vw - 48px));padding:15px 18px;border-radius:18px}.eyebrow{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:750;letter-spacing:.12em;text-transform:uppercase;color:var(--violet)}
.mark{width:9px;height:9px;border-radius:50%;background:linear-gradient(135deg,#776cff,#00bfd2);box-shadow:0 0 0 5px rgba(97,87,235,.10)}
h1{font-size:17px;line-height:1.35;margin:9px 0 3px;font-weight:680;letter-spacing:-.015em}.sub{font-size:12px;color:var(--muted)}
.facts{display:flex;gap:21px;padding:12px 16px;border-radius:16px}.fact span{display:block;font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:#8992a4}.fact strong{display:block;margin-top:3px;font-size:13px}
.transport{position:fixed;z-index:3;left:50%;bottom:22px;transform:translateX(-50%);display:flex;align-items:center;gap:12px;width:min(760px,calc(100vw - 32px));padding:11px 13px;border-radius:18px}
button,select{height:36px;border:1px solid var(--line);border-radius:11px;background:rgba(255,255,255,.86);color:var(--ink);font:600 12px inherit;cursor:pointer;transition:transform .15s,border-color .15s,background .15s}button:hover,select:hover{border-color:#b5bdec;background:#fff}button:active{transform:scale(.97)}
#play{flex:0 0 38px;width:38px;border:0;color:white;background:linear-gradient(135deg,#7065f0,#5147d8);box-shadow:0 7px 18px rgba(81,71,216,.28)}.play-icon{position:relative;display:inline-block;width:12px;height:14px}.play-icon.pause:before,.play-icon.pause:after{content:"";position:absolute;top:1px;width:3px;height:12px;border-radius:2px;background:white}.play-icon.pause:before{left:2px}.play-icon.pause:after{right:2px}.play-icon.play:before{content:"";position:absolute;left:2px;top:1px;border-top:6px solid transparent;border-bottom:6px solid transparent;border-left:10px solid white}
#timeline{appearance:none;flex:1;min-width:80px;height:4px;border:0;border-radius:9px;background:linear-gradient(90deg,var(--violet) var(--played,0%),#dce1eb var(--played,0%));outline:0}#timeline::-webkit-slider-thumb{appearance:none;width:15px;height:15px;border-radius:50%;background:white;border:4px solid var(--violet);box-shadow:0 2px 7px rgba(30,35,60,.22)}#timeline::-moz-range-thumb{width:8px;height:8px;border-radius:50%;background:white;border:4px solid var(--violet)}
#clock{width:76px;text-align:center;font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#4d5669}.toggle.active{color:#5046d7;border-color:#cbc7fa;background:#f0efff}.hint{position:fixed;z-index:2;right:22px;bottom:88px;padding:8px 11px;border-radius:10px;background:rgba(255,255,255,.68);color:#7a8497;font-size:10px;backdrop-filter:blur(10px);pointer-events:none}
#loading{position:fixed;inset:0;z-index:5;display:grid;place-items:center;background:#f6f7fb;color:#677085;font-size:13px;transition:opacity .3s}.spinner{width:24px;height:24px;margin:0 auto 12px;border:2px solid #dfe3ee;border-top-color:var(--violet);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:700px){.facts{display:none}.top{left:14px;right:14px;top:12px}.brand{padding:12px 14px}.transport{bottom:12px;gap:7px}.transport .label{display:none}.hint{display:none}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.185.0/examples/jsm/"}}</script>
</head>
<body>
<main id="stage" aria-label="Interactive 3D motion preview"></main>
<header class="top">
  <section class="brand"><div class="eyebrow"><i class="mark"></i>warp-nn · Kimodo</div><h1 id="prompt"></h1><div class="sub" id="details"></div></section>
  <section class="facts"><div class="fact"><span>Duration</span><strong id="duration"></strong></div><div class="fact"><span>Frames</span><strong id="frames"></strong></div><div class="fact"><span>Seed</span><strong id="seed"></strong></div></section>
</header>
<section class="transport" aria-label="Playback controls">
  <button id="play" title="Play or pause"><span class="play-icon pause"></span></button>
  <span id="clock">0:00.00</span><input id="timeline" type="range" min="0" value="0" step="0.001" aria-label="Timeline">
  <select id="speed" title="Playback speed"><option value=".5">½×</option><option value="1" selected>1×</option><option value="1.5">1½×</option><option value="2">2×</option></select>
  <button id="loop" class="toggle active" title="Loop playback"><span class="label">Loop</span> ↻</button>
  <button id="follow" class="toggle active" title="Follow the character"><span class="label">Follow</span> ◎</button>
  <button id="reset" title="Reset camera"><span class="label">Reset view</span> ⌂</button>
</section>
<div class="hint">Drag to orbit · scroll to zoom · space to play/pause</div>
<div id="loading"><div><div class="spinner"></div><span>Loading the 3D viewer…</span></div></div>
<script>
const PAYLOAD="__PAYLOAD__";
</script>
<script type="module">
const loading=document.querySelector('#loading');
try {
const THREE=await import('three');
const {OrbitControls}=await import('three/addons/controls/OrbitControls.js');
const raw=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(PAYLOAD),c=>c.charCodeAt(0))));
const decode=(text,Type)=>{const bytes=Uint8Array.from(atob(text),c=>c.charCodeAt(0));return new Type(bytes.buffer)};
const positions=decode(raw.positions,Float32Array),contacts=decode(raw.contacts,Uint8Array),meta=raw.meta;
const T=meta.frames,J=meta.joints,FPS=meta.fps,total=(T-1)/FPS,parents=meta.parents;
document.title=`Kimodo · ${meta.prompt}`;document.querySelector('#prompt').textContent=meta.prompt;
document.querySelector('#details').textContent=`${meta.fps} FPS · generated in ${meta.generation_seconds.toFixed(2)}s`;
document.querySelector('#duration').textContent=`${total.toFixed(1)} s`;document.querySelector('#frames').textContent=T;document.querySelector('#seed').textContent=meta.seed;
const stage=document.querySelector('#stage'),scene=new THREE.Scene();scene.background=new THREE.Color(0xf7f8fc);scene.fog=new THREE.Fog(0xf7f8fc,18,42);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;stage.append(renderer.domElement);
const camera=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.03,100);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.075;controls.minDistance=1.6;controls.maxDistance=18;controls.maxPolarAngle=Math.PI*.49;
scene.add(new THREE.HemisphereLight(0xffffff,0xdde2ed,2.15));const sun=new THREE.DirectionalLight(0xffffff,3.3);sun.position.set(-4,8,4);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-7;sun.shadow.camera.right=7;sun.shadow.camera.top=7;sun.shadow.camera.bottom=-7;sun.shadow.bias=-.00015;scene.add(sun);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(80,80),new THREE.ShadowMaterial({color:0x647087,opacity:.13}));ground.rotation.x=-Math.PI/2;ground.receiveShadow=true;scene.add(ground);
const grid=new THREE.GridHelper(80,160,0xb9c1d0,0xdce1ea);grid.material.transparent=true;grid.material.opacity=.42;scene.add(grid);
const jointGeo=new THREE.SphereGeometry(.0367,18,12),boneGeo=new THREE.CylinderGeometry(.027,.027,1,12,1,false);boneGeo.translate(0,.5,0);
const jointMat=new THREE.MeshStandardMaterial({color:0x252a33,roughness:.72,metalness:0});const boneMat=new THREE.MeshStandardMaterial({color:0xd89432,roughness:.58,metalness:.04});
const joints=new THREE.InstancedMesh(jointGeo,jointMat,J),boneCount=parents.filter(x=>x>=0).length,bones=new THREE.InstancedMesh(boneGeo,boneMat,boneCount);joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);joints.castShadow=bones.castShadow=true;scene.add(joints,bones);
const ringGeo=new THREE.RingGeometry(.08,.13,32),ringMat=new THREE.MeshBasicMaterial({color:0x00b8d1,transparent:true,opacity:.78,side:THREE.DoubleSide,depthWrite:false});const rings=[0,1,2,3].map(()=>{const m=new THREE.Mesh(ringGeo,ringMat);m.rotation.x=-Math.PI/2;m.visible=false;scene.add(m);return m});const contactJoints=[24,25,28,29];
const rootPoints=[];for(let f=0;f<T;f++)rootPoints.push(new THREE.Vector3(positions[(f*J)*3],.012,positions[(f*J)*3+2]));if(rootPoints.length>1){const curve=new THREE.CatmullRomCurve3(rootPoints);const trail=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.max(20,T),.008,6,false),new THREE.MeshBasicMaterial({color:0x8d96aa,transparent:true,opacity:.32}));scene.add(trail)}
const dummy=new THREE.Object3D(),a=new THREE.Vector3(),b=new THREE.Vector3(),delta=new THREE.Vector3(),up=new THREE.Vector3(0,1,0),root=new THREE.Vector3(),previousRoot=new THREE.Vector3();
const value=(frame,joint,axis)=>positions[(frame*J+joint)*3+axis];
function sample(frame,joint,out){const i=Math.floor(frame),n=Math.min(i+1,T-1),u=frame-i;out.set(THREE.MathUtils.lerp(value(i,joint,0),value(n,joint,0),u),THREE.MathUtils.lerp(value(i,joint,1),value(n,joint,1),u),THREE.MathUtils.lerp(value(i,joint,2),value(n,joint,2),u));return out}
let elapsed=0,playing=true,looping=true,following=true,speed=1,last=performance.now();
function pose(){const frame=Math.min(T-1,elapsed*FPS);let bi=0;for(let j=0;j<J;j++){sample(frame,j,a);dummy.position.copy(a);dummy.quaternion.identity();dummy.scale.setScalar(j===6?.15:1);dummy.updateMatrix();joints.setMatrixAt(j,dummy.matrix);const p=parents[j];if(p>=0){sample(frame,p,b);delta.subVectors(a,b);dummy.position.copy(b);dummy.quaternion.setFromUnitVectors(up,delta.clone().normalize());dummy.scale.set(1,delta.length(),1);dummy.updateMatrix();bones.setMatrixAt(bi++,dummy.matrix)}}joints.instanceMatrix.needsUpdate=true;bones.instanceMatrix.needsUpdate=true;
sample(frame,0,root);if(following){delta.subVectors(root,previousRoot);camera.position.add(delta);controls.target.add(delta)}previousRoot.copy(root);
const fi=Math.min(T-1,Math.round(frame));for(let i=0;i<4;i++){rings[i].visible=Boolean(contacts[fi*4+i]);if(rings[i].visible){sample(frame,contactJoints[i],a);rings[i].position.set(a.x,.016,a.z)}}
document.querySelector('#timeline').value=elapsed;document.querySelector('#timeline').style.setProperty('--played',`${100*elapsed/Math.max(total,.001)}%`);document.querySelector('#clock').textContent=`${Math.floor(elapsed/60)}:${(elapsed%60).toFixed(2).padStart(5,'0')}`}
function resetCamera(){sample(Math.min(T-1,elapsed*FPS),0,root);controls.target.set(root.x,root.y+.72,root.z);camera.position.set(root.x+3.25,root.y+2.15,root.z+4.6);previousRoot.copy(root);controls.update()}
const timeline=document.querySelector('#timeline');timeline.max=total;timeline.addEventListener('input',()=>{elapsed=Number(timeline.value);playing=false;document.querySelector('#play .icon').textContent='▶';pose()});
const play=document.querySelector('#play');function setPlaying(value){playing=value;play.querySelector('.play-icon').className=`play-icon ${playing?'pause':'play'}`}play.onclick=()=>setPlaying(!playing);document.querySelector('#speed').onchange=e=>speed=Number(e.target.value);
const loop=document.querySelector('#loop');loop.onclick=()=>{looping=!looping;loop.classList.toggle('active',looping)};const follow=document.querySelector('#follow');follow.onclick=()=>{following=!following;follow.classList.toggle('active',following);previousRoot.copy(root)};document.querySelector('#reset').onclick=resetCamera;
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();setPlaying(!playing)}else if(e.code==='ArrowRight'){elapsed=Math.min(total,elapsed+1/FPS);pose()}else if(e.code==='ArrowLeft'){elapsed=Math.max(0,elapsed-1/FPS);pose()}});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
pose();requestAnimationFrame(()=>{resetCamera();pose();renderer.render(scene,camera)});loading.style.opacity=0;setTimeout(()=>loading.remove(),320);
renderer.setAnimationLoop(now=>{const dt=Math.min(.05,(now-last)/1000);last=now;if(playing){elapsed+=dt*speed;if(elapsed>=total){if(looping)elapsed%=Math.max(total,.001);else{elapsed=total;setPlaying(false)}}pose()}controls.update();renderer.render(scene,camera)});
} catch(error) {loading.innerHTML='<div style="max-width:420px;padding:28px;text-align:center"><b>The 3D viewer could not load.</b><br><br><span style="color:#7a8497">This page needs internet access to load Three.js.<br>'+String(error)+'</span></div>';console.error(error)}
</script>
</body>
</html>
"""


def write_motion_html(
    path: str | Path,
    motion,
    *,
    fps: float,
    prompt: str,
    seed: int,
    generation_seconds: float,
) -> Path:
    """Write one Kimodo motion as a single interactive HTML document."""
    positions = np.asarray(motion["posed_joints"], dtype=np.float32)
    contacts = np.asarray(motion["foot_contacts"], dtype=np.uint8)
    if positions.ndim == 4 and positions.shape[0] == 1:
        positions = positions[0]
    if contacts.ndim == 3 and contacts.shape[0] == 1:
        contacts = contacts[0]
    if positions.ndim != 3 or positions.shape[1:] != (30, 3):
        raise ValueError("viewer expects one SOMA-30 motion shaped [frames, 30, 3]")
    if contacts.shape != (positions.shape[0], 4):
        raise ValueError("viewer expects four foot-contact channels per frame")
    if not np.isfinite(positions).all():
        raise ValueError("motion contains non-finite joint positions")

    def encode(value):
        return base64.b64encode(value).decode("ascii")

    payload = {
        "meta": {
            "prompt": str(prompt),
            "seed": int(seed),
            "generation_seconds": float(generation_seconds),
            "fps": float(fps),
            "frames": int(positions.shape[0]),
            "joints": int(positions.shape[1]),
            "parents": _SOMA30_PARENTS.tolist(),
        },
        "positions": encode(np.ascontiguousarray(positions, dtype="<f4").tobytes()),
        "contacts": encode(np.ascontiguousarray(contacts).tobytes()),
    }
    encoded = encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    return destination


__all__ = ["write_motion_html"]
