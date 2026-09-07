# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Write portable, browser-based previews for generated Kimodo motions."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ..skinning import RiggedMesh
from .constraints import SOMA30_PARENTS


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
  <button id="rig" class="toggle" title="Show or hide the skeleton" hidden><span class="label">Rig</span> ◇</button>
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
const T=meta.frames,J=meta.joints,FPS=meta.fps,total=(T-1)/FPS,parents=meta.parents,contactJoints=meta.contact_joints,groundY=meta.ground_y;
document.title=`${meta.label} · ${meta.prompt}`;document.querySelector('.eyebrow').lastChild.textContent=`warp-nn · ${meta.label}`;document.querySelector('#prompt').textContent=meta.prompt;
document.querySelector('#details').textContent=`${meta.fps} FPS · generated in ${meta.generation_seconds.toFixed(2)}s`;
document.querySelector('#duration').textContent=`${total.toFixed(1)} s`;document.querySelector('#frames').textContent=T;document.querySelector('#seed').textContent=meta.seed;
const stage=document.querySelector('#stage'),scene=new THREE.Scene();scene.background=new THREE.Color(0xf7f8fc);scene.fog=new THREE.Fog(0xf7f8fc,18,42);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;stage.append(renderer.domElement);
const camera=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.03,100);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.075;controls.minDistance=.2;controls.maxDistance=50;controls.maxPolarAngle=Math.PI*.49;
scene.add(new THREE.HemisphereLight(0xffffff,0xdde2ed,2.15));const sun=new THREE.DirectionalLight(0xffffff,3.3);sun.position.set(-4,8,4);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-7;sun.shadow.camera.right=7;sun.shadow.camera.top=7;sun.shadow.camera.bottom=-7;sun.shadow.bias=-.00015;scene.add(sun);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(80,80),new THREE.ShadowMaterial({color:0x647087,opacity:.13}));ground.rotation.x=-Math.PI/2;ground.position.y=groundY;ground.receiveShadow=true;scene.add(ground);
const grid=new THREE.GridHelper(80,80,0xb9c1d0,0xdce1ea);grid.position.y=groundY+.001;grid.material.transparent=true;grid.material.opacity=.42;scene.add(grid);
const boneLengths=[];for(let j=0;j<J;j++){const p=parents[j];if(p>=0){const i=j*3,k=p*3;boneLengths.push(Math.hypot(positions[i]-positions[k],positions[i+1]-positions[k+1],positions[i+2]-positions[k+2]))}}boneLengths.sort((x,y)=>x-y);const typical=boneLengths[Math.floor(boneLengths.length/2)]||.15,jointRadius=Math.min(.0245,Math.max(.012,typical*.16));
const jointGeo=new THREE.SphereGeometry(jointRadius,18,12),boneGeo=new THREE.CylinderGeometry(jointRadius*.72,jointRadius*.72,1,12,1,false);boneGeo.translate(0,.5,0);
const jointMat=new THREE.MeshStandardMaterial({color:0x252a33,roughness:.72,metalness:0});const boneMat=new THREE.MeshStandardMaterial({color:0xd89432,roughness:.58,metalness:.04});
const joints=new THREE.InstancedMesh(jointGeo,jointMat,J),boneCount=parents.filter(x=>x>=0).length,bones=new THREE.InstancedMesh(boneGeo,boneMat,boneCount);joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);joints.frustumCulled=bones.frustumCulled=false;joints.castShadow=bones.castShadow=true;scene.add(joints,bones);
let skinUniforms=null,skinRotations=null,skinRest=null;if(raw.mesh){joints.visible=bones.visible=false;const rig=document.querySelector('#rig');rig.hidden=false;rig.onclick=()=>{const visible=!joints.visible;joints.visible=bones.visible=visible;rig.classList.toggle('active',visible)};const m=raw.mesh,vertices=decode(m.vertices,Float32Array),faces=decode(m.faces,Uint32Array),skinJoint=decode(m.joints,Uint16Array),skinWeight=decode(m.weights,Float32Array);skinRotations=decode(m.rotations,Float32Array);skinRest=decode(m.restJoints,Float32Array);const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(vertices,3));geometry.setAttribute('skinJoint',new THREE.BufferAttribute(skinJoint,4));geometry.setAttribute('skinWeight',new THREE.BufferAttribute(skinWeight,4));geometry.setIndex(new THREE.BufferAttribute(faces,1));geometry.computeVertexNormals();skinUniforms={rigMatrices:{value:Array.from({length:J},()=>new THREE.Matrix4())}};const applySkinning=shader=>{Object.assign(shader.uniforms,skinUniforms);shader.vertexShader=`attribute vec4 skinJoint;\nattribute vec4 skinWeight;\nuniform mat4 rigMatrices[${J}];\n`+shader.vertexShader;shader.vertexShader=shader.vertexShader.replace('#include <beginnormal_vertex>','mat3 skinRotation = mat3(rigMatrices[int(skinJoint.x)]) * skinWeight.x + mat3(rigMatrices[int(skinJoint.y)]) * skinWeight.y + mat3(rigMatrices[int(skinJoint.z)]) * skinWeight.z + mat3(rigMatrices[int(skinJoint.w)]) * skinWeight.w;\nvec3 objectNormal = skinRotation * normal;');shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>','vec4 skinned = (rigMatrices[int(skinJoint.x)] * vec4(position,1.0)) * skinWeight.x + (rigMatrices[int(skinJoint.y)] * vec4(position,1.0)) * skinWeight.y + (rigMatrices[int(skinJoint.z)] * vec4(position,1.0)) * skinWeight.z + (rigMatrices[int(skinJoint.w)] * vec4(position,1.0)) * skinWeight.w;\nvec3 transformed = skinned.xyz;')};const material=new THREE.MeshStandardMaterial({color:0x6faee8,roughness:.76,metalness:0,side:THREE.DoubleSide});material.onBeforeCompile=applySkinning;const depthMaterial=new THREE.MeshDepthMaterial({depthPacking:THREE.RGBADepthPacking,side:THREE.DoubleSide});depthMaterial.onBeforeCompile=applySkinning;const body=new THREE.Mesh(geometry,material);body.customDepthMaterial=depthMaterial;body.castShadow=body.receiveShadow=true;body.frustumCulled=false;scene.add(body)}
const ringGeo=new THREE.RingGeometry(jointRadius*2.3,jointRadius*3.6,32),ringMat=new THREE.MeshBasicMaterial({color:0x00b8d1,transparent:true,opacity:.78,side:THREE.DoubleSide,depthWrite:false});const rings=contactJoints.map(()=>{const m=new THREE.Mesh(ringGeo,ringMat);m.rotation.x=-Math.PI/2;m.visible=false;scene.add(m);return m});
const rootPoints=[];for(let f=0;f<T;f++)rootPoints.push(new THREE.Vector3(positions[(f*J)*3],groundY+.012,positions[(f*J)*3+2]));if(rootPoints.length>1){const curve=new THREE.CatmullRomCurve3(rootPoints);const trail=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.max(20,T),.008,6,false),new THREE.MeshBasicMaterial({color:0x8d96aa,transparent:true,opacity:.32}));scene.add(trail)}
const dummy=new THREE.Object3D(),a=new THREE.Vector3(),b=new THREE.Vector3(),delta=new THREE.Vector3(),up=new THREE.Vector3(0,1,0),root=new THREE.Vector3(),focus=new THREE.Vector3(),previousFocus=new THREE.Vector3(),skinMatrixA=new THREE.Matrix4(),skinMatrixB=new THREE.Matrix4(),skinQuaternionA=new THREE.Quaternion(),skinQuaternionB=new THREE.Quaternion(),skinQuaternion=new THREE.Quaternion();
const value=(frame,joint,axis)=>positions[(frame*J+joint)*3+axis];
function sample(frame,joint,out){const i=Math.floor(frame),n=Math.min(i+1,T-1),u=frame-i;out.set(THREE.MathUtils.lerp(value(i,joint,0),value(n,joint,0),u),THREE.MathUtils.lerp(value(i,joint,1),value(n,joint,1),u),THREE.MathUtils.lerp(value(i,joint,2),value(n,joint,2),u));return out}
let elapsed=0,playing=T>2,looping=true,following=true,ready=false,speed=1,last=performance.now();if(!playing)document.querySelector('#play').querySelector('.play-icon').className='play-icon play';
function pose(){const frame=Math.min(T-1,elapsed*FPS);let bi=0,minY=Infinity,maxY=-Infinity;for(let j=0;j<J;j++){sample(frame,j,a);minY=Math.min(minY,a.y);maxY=Math.max(maxY,a.y);const facial=J===30&&j>=7&&j<=9,markerScale=facial?.42:1;dummy.position.copy(a);dummy.quaternion.identity();dummy.scale.setScalar(markerScale);dummy.updateMatrix();joints.setMatrixAt(j,dummy.matrix);const p=parents[j];if(p>=0){sample(frame,p,b);delta.subVectors(a,b);dummy.position.copy(b);dummy.quaternion.setFromUnitVectors(up,delta.clone().normalize());dummy.scale.set(markerScale,delta.length(),markerScale);dummy.updateMatrix();bones.setMatrixAt(bi++,dummy.matrix)}}joints.instanceMatrix.needsUpdate=true;bones.instanceMatrix.needsUpdate=true;if(skinUniforms){const first=Math.floor(frame),next=Math.min(first+1,T-1),amount=frame-first,rot=skinRotations,rest=skinRest;for(let j=0;j<J;j++){const ia=(first*J+j)*9,ib=(next*J+j)*9,rj=j*3;skinMatrixA.set(rot[ia],rot[ia+1],rot[ia+2],0,rot[ia+3],rot[ia+4],rot[ia+5],0,rot[ia+6],rot[ia+7],rot[ia+8],0,0,0,0,1);skinMatrixB.set(rot[ib],rot[ib+1],rot[ib+2],0,rot[ib+3],rot[ib+4],rot[ib+5],0,rot[ib+6],rot[ib+7],rot[ib+8],0,0,0,0,1);skinQuaternionA.setFromRotationMatrix(skinMatrixA);skinQuaternionB.setFromRotationMatrix(skinMatrixB);skinQuaternion.slerpQuaternions(skinQuaternionA,skinQuaternionB,amount);const matrix=skinUniforms.rigMatrices.value[j];matrix.makeRotationFromQuaternion(skinQuaternion);const e=matrix.elements,x=rest[rj],y=rest[rj+1],z=rest[rj+2];sample(frame,j,a);e[12]=a.x-e[0]*x-e[4]*y-e[8]*z;e[13]=a.y-e[1]*x-e[5]*y-e[9]*z;e[14]=a.z-e[2]*x-e[6]*y-e[10]*z}}
sample(frame,0,root);focus.set(root.x,(minY+maxY)/2,root.z);if(following&&ready){delta.subVectors(focus,previousFocus);camera.position.add(delta);controls.target.add(delta)}previousFocus.copy(focus);
const fi=Math.min(T-1,Math.round(frame)),C=contactJoints.length;for(let i=0;i<C;i++){rings[i].visible=Boolean(contacts[fi*C+i]);if(rings[i].visible){sample(frame,contactJoints[i],a);rings[i].position.set(a.x,groundY+.016,a.z)}}
document.querySelector('#timeline').value=elapsed;document.querySelector('#timeline').style.setProperty('--played',`${100*elapsed/Math.max(total,.001)}%`);document.querySelector('#clock').textContent=`${Math.floor(elapsed/60)}:${(elapsed%60).toFixed(2).padStart(5,'0')}`}
function resetCamera(){const frame=Math.min(T-1,elapsed*FPS);sample(frame,0,root);let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;for(let j=0;j<J;j++){sample(frame,j,a);minX=Math.min(minX,a.x);minY=Math.min(minY,a.y);minZ=Math.min(minZ,a.z);maxX=Math.max(maxX,a.x);maxY=Math.max(maxY,a.y);maxZ=Math.max(maxZ,a.z)}const centerY=(minY+maxY)/2,span=Math.max(maxX-minX,maxY-minY,maxZ-minZ,.4),distance=span*2.2;focus.set(root.x,centerY,root.z);controls.target.copy(focus);camera.position.set(root.x+distance*.58,centerY+distance*.3,root.z+distance*.88);previousFocus.copy(focus);controls.update()}
const timeline=document.querySelector('#timeline');timeline.max=total;timeline.addEventListener('input',()=>{elapsed=Number(timeline.value);setPlaying(false);pose()});
const play=document.querySelector('#play');function setPlaying(value){playing=value;play.querySelector('.play-icon').className=`play-icon ${playing?'pause':'play'}`}play.onclick=()=>setPlaying(!playing);document.querySelector('#speed').onchange=e=>speed=Number(e.target.value);
const loop=document.querySelector('#loop');loop.onclick=()=>{looping=!looping;loop.classList.toggle('active',looping)};const follow=document.querySelector('#follow');follow.onclick=()=>{following=!following;follow.classList.toggle('active',following);previousFocus.copy(focus)};document.querySelector('#reset').onclick=resetCamera;
addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();setPlaying(!playing)}else if(e.code==='ArrowRight'){elapsed=Math.min(total,elapsed+1/FPS);pose()}else if(e.code==='ArrowLeft'){elapsed=Math.max(0,elapsed-1/FPS);pose()}});
addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
pose();renderer.setAnimationLoop(now=>{if(!ready){resetCamera();pose();ready=true;last=now;loading.style.opacity=0;setTimeout(()=>{loading.remove();requestAnimationFrame(()=>{resetCamera();pose()})},320)}else{const dt=Math.min(.05,(now-last)/1000);last=now;if(playing){elapsed+=dt*speed;if(elapsed>=total){if(looping)elapsed%=Math.max(total,.001);else{elapsed=total;setPlaying(false)}}pose()}}controls.update();renderer.render(scene,camera)});
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
    mesh: RiggedMesh | None = None,
    label: str = "Kimodo",
) -> Path:
    """Write one articulated motion as a single interactive HTML document."""
    positions = np.asarray(motion["posed_joints"], dtype=np.float32)
    contacts = np.asarray(motion["foot_contacts"], dtype=np.uint8)
    if positions.ndim == 4 and positions.shape[0] == 1:
        positions = positions[0]
    if contacts.ndim == 3 and contacts.shape[0] == 1:
        contacts = contacts[0]
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("viewer expects one motion shaped [frames, joints, 3]")
    parents = np.asarray(motion.get("parents", SOMA30_PARENTS), dtype=np.int32)
    if parents.shape != (positions.shape[1],):
        raise ValueError("motion parents must contain one entry per joint")
    default_contacts = (24, 25, 28, 29) if positions.shape[1] == 30 else ()
    contact_joints = np.asarray(
        motion.get("contact_joints", default_contacts), dtype=np.int32
    )
    if contacts.shape != (positions.shape[0], len(contact_joints)):
        raise ValueError("foot contacts must contain one channel per contact joint")
    if np.any(contact_joints < 0) or np.any(contact_joints >= positions.shape[1]):
        raise ValueError("contact joint index is out of range")
    if not np.isfinite(positions).all():
        raise ValueError("motion contains non-finite joint positions")

    def encode(value):
        return base64.b64encode(value).decode("ascii")

    ground_y = float(np.min(positions[..., 1]))
    if mesh is not None:
        ground_y += float(np.min(mesh.vertices[:, 1]) - np.min(mesh.rest_joints[:, 1]))
    payload = {
        "meta": {
            "prompt": str(prompt),
            "seed": int(seed),
            "generation_seconds": float(generation_seconds),
            "fps": float(fps),
            "frames": int(positions.shape[0]),
            "joints": int(positions.shape[1]),
            "parents": parents.tolist(),
            "contact_joints": contact_joints.tolist(),
            "joint_names": list(motion.get("joint_names", ())),
            "label": str(label),
            "ground_y": ground_y,
        },
        "positions": encode(np.ascontiguousarray(positions, dtype="<f4").tobytes()),
        "contacts": encode(np.ascontiguousarray(contacts).tobytes()),
    }
    if mesh is not None:
        if len(mesh.rest_joints) != len(parents) or not np.array_equal(
            mesh.parents, parents
        ):
            raise ValueError("mesh and motion must use the same ordered hierarchy")
        rotations = np.asarray(motion["global_rot_mats"], dtype=np.float32)
        if rotations.ndim == 5 and rotations.shape[0] == 1:
            rotations = rotations[0]
        if rotations.shape != (len(positions), len(parents), 3, 3):
            raise ValueError("mesh preview requires global_rot_mats per frame")
        joint_indices, joint_weights = mesh.top4
        payload["mesh"] = {
            "vertices": encode(mesh.vertices.astype("<f4", copy=False).tobytes()),
            "faces": encode(mesh.faces.astype("<u4", copy=False).tobytes()),
            "restJoints": encode(mesh.rest_joints.astype("<f4", copy=False).tobytes()),
            "joints": encode(joint_indices.astype("<u2", copy=False).tobytes()),
            "weights": encode(joint_weights.astype("<f4", copy=False).tobytes()),
            "rotations": encode(np.ascontiguousarray(rotations, dtype="<f4").tobytes()),
        }
    encoded = encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    return destination


__all__ = ["write_motion_html"]
