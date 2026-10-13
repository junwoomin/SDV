export function diagnosticsHTML(){return `<section class="panel" id="system-diagnostics"><div class="panel-head"><div><h2>System Diagnostics · CARLA</h2><p>실제 GPU · Vulkan · 엔진 상태를 검사합니다. 최소 엔진 검사는 센서 수집 검증과 별도입니다.</p></div></div><div class="panel-body"><div class="button-row"><button type="button" class="button subtle" id="diag-run">Run Diagnostics</button><button type="button" class="button primary" id="diag-test">Test CARLA Engine</button><button type="button" class="button subtle" id="diag-logs">View Engine Logs</button><a class="button subtle" href="/api/carla/diagnostics/export" download>Export Diagnostics</a></div><p id="diag-message" role="status">Run Diagnostics로 현재 시스템을 확인하세요.</p><div id="diag-results"></div><details><summary>상세 환경 / 엔진 로그</summary><pre id="diag-log" class="log-output" style="white-space:pre-wrap;max-height:420px;overflow:auto"></pre></details></div></section>`;}
export function mountDiagnostics({api}){
 const root=document.getElementById('system-diagnostics');if(!root)return;
 let busy=false;
 const message=root.querySelector('#diag-message'),log=root.querySelector('#diag-log'),results=root.querySelector('#diag-results');
 const esc=value=>String(value??'N/A').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 async function perform(action){if(busy)return;busy=true;root.querySelectorAll('button').forEach(b=>b.disabled=true);message.textContent='검사 중…';
  try{
   if(action==='test'){const r=await api('/api/carla/diagnostics/test','POST');message.textContent=r.task?.message||'최소 테스트 시작. Run Diagnostics 또는 엔진 로그에서 완료 결과를 확인하세요.';return;}
   if(action==='logs'){const r=await api('/api/carla/local/logs');log.textContent=r.text;root.querySelector('details').open=true;message.textContent='현재 실행의 엔진·Unreal 로그입니다.';return;}
   const r=await api('/api/carla/diagnostics','POST'),gpu=r.gpu?.[0]||{},engine=r.engine||{};
   const rows={'CARLA / Unreal':`${r.carla_version||'미설치'} / ${r.unreal_version}`,'GPU / Driver':`${gpu.name||'조회 실패'} / ${gpu.driver_version||'N/A'}`,'Vulkan':(r.vulkan?.devices||[]).map(d=>`${d.name} (${d.api_version})`).join(', ')||r.vulkan?.error,'VRAM MiB':`${gpu.memory_used_mib||'N/A'} / ${gpu.memory_total_mib||'N/A'}`,'CARLA PID':engine.pid,'Engine state':engine.state||'Stopped','RPC Port':r.rpc_port,'Engine Exit Code':engine.exit_code,'실행 / 다음 프로파일':`${engine.profile||'N/A'} / ${r.render_profile}`,'마지막 단계':engine.last_stage,'마지막 오류':engine.last_error||(r.task?.status==='Failed'?r.task?.message:null)};
   results.innerHTML=`<dl class="keyvalue">${Object.entries(rows).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`;
   message.textContent=r.warnings?.length?r.warnings.join(' · '):'GPU/Vulkan 환경 검사 완료. 수집 가능 여부는 실제 수집 검사로 확인하세요.';
   log.textContent=JSON.stringify({...r,logs:undefined},null,2)+'\n\n'+(r.logs||'');
  }catch(error){message.textContent=error.message;}
  finally{busy=false;root.querySelectorAll('button').forEach(b=>b.disabled=false);}
 }
 root.querySelector('#diag-run').onclick=()=>perform('run');root.querySelector('#diag-test').onclick=()=>perform('test');root.querySelector('#diag-logs').onclick=()=>perform('logs');
}
