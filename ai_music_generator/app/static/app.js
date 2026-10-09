const $=id=>document.getElementById(id);let buf=null,raf=0;const audio=$("audio");
function setState(t,c=""){const s=$("state");s.textContent=t;s.className="state "+c}
fetch("/api/status").then(r=>r.json()).then(s=>{const m=$("model");
 if(s.trained){m.textContent=`Model ready · ${s.windows} windows · loss ${s.final_loss}`;m.className="pill ok";
  $("genre").innerHTML=s.genres.map(g=>`<option>${g}</option>`).join("")}
 else{m.textContent="Model not trained";m.className="pill bad";$("gen").disabled=true;setState(s.message,"error")}});
$("bpm").oninput=e=>$("bpmv").textContent=e.target.value;
function draw(p=0){const c=$("wave"),w=c.width=c.clientWidth,h=c.height=c.clientHeight,x=c.getContext("2d");x.clearRect(0,0,w,h);
 if(!buf)return;const d=buf.getChannelData(0),step=Math.floor(d.length/w);x.fillStyle="#7c5cff";
 for(let i=0;i<w;i++){let mx=0;for(let j=0;j<step;j+=20)mx=Math.max(mx,Math.abs(d[i*step+j]||0));x.fillRect(i,h/2-mx*h/2,1,mx*h+1)}
 x.fillStyle="#ff4d8d";x.fillRect(p*w,0,2,h)}
function tick(){draw(audio.currentTime/(audio.duration||1));raf=requestAnimationFrame(tick)}
$("play").onclick=()=>{if(audio.paused){audio.play();$("play").textContent="❚❚ Pause";tick()}else{audio.pause();$("play").textContent="▶ Play";cancelAnimationFrame(raf)}};
$("stop").onclick=()=>{audio.pause();audio.currentTime=0;$("play").textContent="▶ Play";cancelAnimationFrame(raf);draw(0)};
audio.onended=()=>{$("play").textContent="▶ Play";cancelAnimationFrame(raf)};
$("gen").onclick=async()=>{$("gen").disabled=true;["play","stop"].forEach(i=>$(i).disabled=true);setState("Generating with the LSTM… this can take up to a minute on CPU.","load");
 try{const r=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
  genre:$("genre").value,mood:$("mood").value,instrument:$("instrument").value,bpm:+$("bpm").value,duration:+$("duration").value,start_notes:$("notes").value,text:$("text").value})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Generation failed");
  audio.src=d.wav;$("dlmid").href=d.midi;$("dlwav").href=d.wav;$("dlmid").classList.remove("disabled");$("dlwav").classList.remove("disabled");
  const ab=await (await fetch(d.wav)).arrayBuffer();buf=await new (window.AudioContext||window.webkitAudioContext)().decodeAudioData(ab);draw(0);
  $("play").disabled=$("stop").disabled=false;$("play").textContent="▶ Play";
  $("info").innerHTML=`${d.genre} · ${d.mood} · ${d.instrument} · ${d.bpm} BPM · ${d.seconds}s · ${d.notes} notes · seed notes ${d.seed_used?"used":"not given"}`+d.warnings.map(w=>`<br>⚠ ${w}`).join("");
  setState("Done — track generated.","ok")}
 catch(e){setState("Error: "+e.message,"error")}$("gen").disabled=false};
