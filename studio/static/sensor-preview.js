/* Live CARLA preview controls. Mount diagrams are explicitly schematic.
 * Integration: SensorPreview.html(); mount({state, api, imageSource, confirm});
 * update(state) accepts the normal Studio state, including sensor_preview.
 * Navigation only stops browser polling; stop is an explicit server action.
 */
(() => {
  'use strict';
  const text = {
    '센서 배치와 실제 프리뷰':'Sensor layout and live preview',
    '저장 전 센서 위치와 시야를 CARLA에서 확인합니다.':'Check sensor positions and views in CARLA before saving.',
    '프리뷰 시작':'Start preview','프리뷰 중지':'Stop preview','프리뷰에 적용':'Apply to preview','실행 로그':'Run logs',
    '센서 선택':'Select sensor','센서 영상':'Sensor image',
    '대기':'Idle','실행 중':'Running','중지됨':'Stopped','실패':'Failed','완료':'Completed',
    '서버 확인 중':'Checking server','연결 중':'Connecting','영상 수신 중':'Receiving frames','설정 적용 중':'Applying configuration','정리 중':'Cleaning up','중지 요청 중':'Stopping',
    '실제 프레임이 아직 없습니다.':'No actual frame is available yet.',
    'CARLA를 연결하고 프리뷰를 시작하세요.':'Connect CARLA and start the preview.',
    '이 센서를 활성화한 뒤 프리뷰에 적용하세요.':'Enable this sensor, then apply it to the preview.',
    'LiDAR 포인트를 영상으로 투영한 실제 센서 프레임입니다.':'A visualization of actual LiDAR points.',
    'CARLA 센서가 생성한 RGB 영상입니다.':'RGB image produced by the CARLA sensor.',
    '촬영 시각':'Captured at','프레임':'Frame','프레임 시각 없음':'Frame timestamp unavailable','최신 프레임 대기':'Waiting for a new frame',
    '마지막 수신 프레임':'Last received frame','저장 전 배치도':'Layout draft','위에서 본 배치':'Top layout','옆에서 본 배치':'Side layout',
    '배치도는 실제 영상이 아닙니다. 점을 끌거나 빈 위치를 눌러 선택 센서를 이동하세요.':'These diagrams are not camera images. Drag a marker or click a position to move the selected sensor.',
    '점은 센서의 대략적인 위치를 나타냅니다.':'The dots show the approximate positions of the sensors.',
    '다음 단계의 준비 조건':'Requirements for the next step',
    '실제 센서를 확인하려면 프리뷰를 시작하세요.':'Start the preview to check the actual sensors.',
    '변경한 설정을 프리뷰에 적용하세요.':'Apply the changed configuration to the preview.',
    '선택 센서의 실제 프레임 수신을 기다리세요.':'Wait for an actual frame from the selected sensor.',
    '센서 준비가 끝날 때까지 기다리세요.':'Wait until the sensors are ready.',
    '선택 센서를 활성화하고 프리뷰에 적용하세요.':'Enable the selected sensor and apply it to the preview.',
    '마지막 상태 확인 기준':'At the last status check',
    '화면 갱신 상한':'Display refresh limit','설정한 목표 FPS':'Configured target FPS',
    '현재 센서 요청 FPS':'Current sensor target FPS','실제 수신 FPS':'Measured source FPS','실제 출력 FPS':'Measured output FPS',
    '저장한 FPS를 프리뷰에 적용하세요.':'Apply the saved FPS to the preview.',
    '실제 FPS는 CARLA 실행 속도에 따라 낮아질 수 있습니다.':'Actual FPS may be lower depending on CARLA execution speed.',
    '키보드 방향키: 0.1 m 이동 · Shift: 0.5 m':'Arrow keys: move 0.1 m · Shift: 0.5 m',
    '차량 기준 좌표':'Vehicle coordinates','앞':'Front','뒤':'Rear','왼쪽':'Left','오른쪽':'Right','원점':'Origin',
    '전방 카메라':'Front camera','후방 카메라':'Rear camera','좌측 카메라':'Left camera','우측 카메라':'Right camera',
    '전방 좌측':'Front left','전방 우측':'Front right','후방 좌측':'Rear left','후방 우측':'Rear right','전방 고해상도':'Front high resolution',
    '비활성':'Disabled','수치 설정으로 이동':'Open numeric settings',
    '프리뷰 적용은 영구 저장과 별개입니다. 최종 구성은 아래의 센서 설정 저장으로 보존하세요.':'Applying a preview does not save the configuration. Use Save sensor configuration below to keep the final setup.',
    '프리뷰에 적용하지 않은 변경이 있습니다.':'The preview has unapplied changes.',
    '프리뷰에 현재 설정이 적용되었습니다.':'The current configuration is applied to the preview.',
    '프리뷰 설정을 전송했습니다. 새 프레임을 기다립니다.':'Preview configuration sent. Waiting for a new frame.',
    '서버에 연결할 수 없습니다. 연결 상태를 확인하세요.':'Cannot reach the server. Check the connection.',
    '연구 작업이 실행 중입니다. 작업이 끝난 뒤 프리뷰를 시작하세요.':'A research job is running. Start the preview after it finishes.',
    'CARLA 설치·기동이 진행 중입니다. 준비가 끝날 때까지 기다리세요.':'CARLA installation or startup is in progress. Wait until it is ready.',
    'CARLA 서버를 먼저 연결하세요.':'Connect the CARLA server first.',
    '활성 카메라를 하나 이상 선택하세요.':'Select at least one active camera.',
    '프리뷰가 이미 실행 중입니다.':'The preview is already running.',
    '프리뷰를 시작한 뒤 설정을 적용하세요.':'Start the preview before applying configuration.',
    '프리뷰 준비·정리 중에는 설정을 적용할 수 없습니다.':'Configuration cannot be applied while the preview is starting or cleaning up.',
    'CARLA 설치·연결로 이동':'Open CARLA setup and connection',
    '실제 센서 프리뷰를 시작할까요?':'Start a live sensor preview?',
    '연결된 CARLA에 프리뷰용 차량과 센서를 생성합니다. 수집 데이터셋에는 저장하지 않습니다.':'Creates a preview vehicle and sensors in the connected CARLA world. Frames are not added to the collection dataset.',
    '프리뷰를 중지할까요?':'Stop the preview?',
    '프리뷰 차량과 센서를 정리합니다. 저장한 센서 구성은 유지됩니다.':'Removes the preview vehicle and sensors. Saved sensor configuration is retained.',
    '센서 폼이 아직 준비되지 않았습니다.':'The sensor form is not ready yet.',
    '입력한 센서 수치를 확인하세요.':'Check the entered sensor values.',
    '기록된 로그가 없습니다.':'No logs have been recorded.',
    '로그를 불러오는 중입니다.':'Loading logs…','로그 새로고침':'Refresh logs',
    '프리뷰 상태를 불러올 수 없습니다.':'Unable to load preview status.',
    '선택 센서':'Selected sensor','활성 센서':'Active sensors','센서':'Sensor','RGB 프리뷰':'RGB preview',
    '수집 채널의 Seg/Depth/OD 설정은 유지되며 이 화면에서는 RGB만 확인합니다.':'Collection Seg/Depth/OD settings are retained; this preview shows RGB only.',
    '선택 센서를 활성화하려면 아래 센서 구성에서 수집에 사용을 선택하세요.':'To enable this sensor, select Use for collection in the sensor configuration below.',
    '원본 서버 메시지':'Original server message','시뮬레이션 시각':'Simulation time','요청 처리 중':'Request in progress','새 프레임을 불러오지 못했습니다.':'Unable to load the new frame.'
  };
  const names = {front_cam:'전방 카메라',rear_cam:'후방 카메라',right_cam:'우측 카메라',left_cam:'좌측 카메라',rear_right_cam:'후방 우측',rear_left_cam:'후방 좌측',front_cam_pv:'전방 고해상도',lidar:'LiDAR'};
  const t = value => window.SDVI18n?.language === 'en' ? (text[value] || window.SDVI18n?.t(value) || value) : value;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clone = value => JSON.parse(JSON.stringify(value));
  const sensorName = id => t(names[id] || id);
  const diagnostic = value => window.SDVI18n?.language==='en'&&/[가-힣]/.test(String(value||''))?`${t('원본 서버 메시지')}: ${value}`:String(value||'');
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const fmt = v => finite(Number(v)) ? Number(v).toFixed(2).replace(/\.00$/, '') : '—';
  const phaseNames = {idle:'대기',starting:'서버 확인 중',connecting:'연결 중',live:'영상 수신 중',applying:'설정 적용 중',stopping:'중지 요청 중',cleaning:'정리 중',failed:'실패',stopped:'중지됨'};
  const statusNames = {Idle:'대기',Running:'실행 중',Completed:'완료',Failed:'실패',Stopped:'중지됨'};
  let root = null, hooks = {}, state = {}, preview = {}, timer = null, frameTimer = null, selected = 'front_cam';
  let lastStatus = 0, statusPending = false, requestPending = false, applyPending = false, dirty = false, generation = 0, noticeError = '', statusError = '';
  let form = null, formListener = null, visibilityListener = null, renderScheduled = false, drag = null, lastImageRequest = 0;
  const loaded = {}, pending = {}, imageTokens = {}, imageRequests = {}, mapTransforms = {};
  const $ = selector => root?.querySelector(selector);
  const stamp = value => {
    if (value === undefined || value === null || value === '') return t('프레임 시각 없음');
    const date = new Date(typeof value === 'number' && value < 1e11 ? value * 1000 : value);
    if (Number.isNaN(date.getTime())) return t('프레임 시각 없음');
    return date.toLocaleTimeString(window.SDVI18n?.language === 'en' ? 'en-US' : 'ko-KR', {hour12:false});
  };
  function html() {
    return `<section id="sensor-preview-control" class="panel sp-panel" aria-labelledby="sp-title">
      <div class="panel-head sp-head"><div><h2 id="sp-title">${t('센서 배치와 실제 프리뷰')}</h2><p>${t('저장 전 센서 위치와 시야를 CARLA에서 확인합니다.')}</p></div><span id="sp-status" class="badge Idle">${t('대기')}</span></div>
      <div class="sp-toolbar"><div class="sp-picker"><label for="sp-sensor">${t('센서 선택')}</label><select id="sp-sensor" class="select-control"></select></div><div class="sp-actions"><button type="button" class="button primary" data-sp="start">${t('프리뷰 시작')}</button><button type="button" class="button" data-sp="apply">${t('프리뷰에 적용')}</button><button type="button" class="button danger" data-sp="stop">${t('프리뷰 중지')}</button></div></div>
      <div class="sp-feedback" id="sp-blocker" role="status"></div><div class="sp-feedback" id="sp-notice" aria-live="polite"></div>
      <div class="sp-views"><figure class="sp-figure"><div class="sp-frame-heading"><strong id="sp-camera-label">${t('센서 영상')}</strong><span id="sp-channel-label">RGB</span></div><div class="sp-image-stage" data-sp-stage="sensor"><div class="sp-no-frame"><span class="sp-reticle" aria-hidden="true"></span><strong>${t('실제 프레임이 아직 없습니다.')}</strong><p>${t('CARLA를 연결하고 프리뷰를 시작하세요.')}</p></div></div><figcaption><span data-sp-frame="sensor">${t('프레임')} —</span><span data-sp-time="sensor">${t('촬영 시각')} —</span><span data-sp-fresh="sensor"></span></figcaption></figure>
      </div>
      <p class="sp-rate" id="sp-rate"></p>
      <div class="sp-layout"><div class="sp-layout-heading"><div><h3>${t('저장 전 배치도')}</h3><p>${t('배치도는 실제 영상이 아닙니다. 점을 끌거나 빈 위치를 눌러 선택 센서를 이동하세요.')} ${t('점은 센서의 대략적인 위치를 나타냅니다.')}</p></div><button type="button" class="text-link" data-sp="edit">${t('수치 설정으로 이동')}</button></div><div class="sp-diagrams"><div class="sp-diagram"><span>${t('위에서 본 배치')} <small>X / Y · m</small></span><svg data-sp-map="top" viewBox="0 0 420 220" tabindex="0" role="application" aria-label="${t('위에서 본 배치')}. ${t('키보드 방향키: 0.1 m 이동 · Shift: 0.5 m')}"></svg></div><div class="sp-diagram"><span>${t('옆에서 본 배치')} <small>X / Z · m</small></span><svg data-sp-map="side" viewBox="0 0 420 220" tabindex="0" role="application" aria-label="${t('옆에서 본 배치')}. ${t('키보드 방향키: 0.1 m 이동 · Shift: 0.5 m')}"></svg></div></div>
      <div class="sp-pose"><strong id="sp-pose-label">${t('차량 기준 좌표')}</strong><div class="sp-pose-fields">${['x','y','z','yaw','pitch','roll'].map(key=>`<label for="sp-${key}"><span>${key.toUpperCase()} <small>${['x','y','z'].includes(key)?'m':'°'}</small></span><input id="sp-${key}" data-sp-pose="${key}" type="number" step="${['x','y','z'].includes(key)?'.1':'1'}" min="${key==='z'?'.1':['x','y'].includes(key)?'-10':'-180'}" max="${['x','y','z'].includes(key)?'10':'180'}"></label>`).join('')}</div></div><p class="sp-hint">${t('키보드 방향키: 0.1 m 이동 · Shift: 0.5 m')}</p></div>
      <div class="panel-foot sp-foot"><span>${t('프리뷰 적용은 영구 저장과 별개입니다. 최종 구성은 아래의 센서 설정 저장으로 보존하세요.')}</span><button type="button" class="text-link" data-sp="logs">${t('실행 로그')}</button></div><details class="sp-logs"><summary>${t('실행 로그')}</summary><div><button type="button" class="button subtle small" data-sp="logs">${t('로그 새로고침')}</button><pre id="sp-logs">${t('기록된 로그가 없습니다.')}</pre></div></details>
    </section>`;
  }
  function readDraft() {
    if (hooks.readDraft) return hooks.readDraft();
    const current = form || document.querySelector('#sensor-form');
    if (!current) throw new Error(t('센서 폼이 아직 준비되지 않았습니다.'));
    const schema = clone(state.sensors || {towns:['Town01'],data_num:100,sensors:{}});
    schema.sensors = {};
    for (const checkbox of current.querySelectorAll('input[name^="enable_"]')) {
      if (!checkbox.checked) continue;
      const name = checkbox.name.slice(7), fields = {};
      for (const input of current.querySelectorAll('input[name]')) {
        if (!input.name.startsWith(name+'.')) continue;
        const key = input.name.slice(name.length+1);
        if (input.type === 'checkbox') fields[key] = input.checked;
        else {
          if (input.value.trim() === '' || !Number.isFinite(Number(input.value))) throw new Error(`${sensorName(name)} · ${key}: ${t('입력한 센서 수치를 확인하세요.')}`);
          fields[key] = Number(input.value);
        }
      }
      schema.sensors[name] = fields;
    }
    return schema;
  }
  function safeDraft() {try {return readDraft();} catch {return {sensors:{}};}}
  function pose(name = selected) {
    const value = {};
    for (const key of ['x','y','z','yaw','pitch','roll']) {
      const input = form?.elements.namedItem(name+'.'+key);
      const fallback = state.sensors?.sensors?.[name]?.[key] ?? 0;
      value[key] = input && input.value.trim() !== '' && Number.isFinite(Number(input.value)) ? Number(input.value) : fallback;
    }
    return value;
  }
  function writePose(values) {
    if (!safeDraft().sensors?.[selected]) return;
    if (hooks.writeMount) hooks.writeMount(selected, values);
    else for (const [key, value] of Object.entries(values)) {
      const input = form?.elements.namedItem(selected+'.'+key);
      if (!input || !Number.isFinite(value)) continue;
      input.value = String(Math.round(value*100)/100);
      input.dispatchEvent(new Event('input', {bubbles:true}));
    }
    dirty = true;noticeError='';
    renderDraft(); renderStatus();
  }
  function renderDraft() {
    if (!root?.isConnected) return;
    const draft = safeDraft(), enabled = Object.keys(draft.sensors || {});
    const configured = [...(form?.querySelectorAll('input[name^="enable_"]') || [])].map(e=>e.name.slice(7));
    const choices = [...new Set(configured.length?configured:Object.keys(state.sensors?.sensors||{}))];
    const picker = $('#sp-sensor');
    const signature = choices.map(id=>id+':'+enabled.includes(id)).join('|');
    if (picker.dataset.signature !== signature) {
      picker.innerHTML = choices.map(id=>`<option value="${esc(id)}">${esc(sensorName(id))}${enabled.includes(id)?'':` (${t('비활성')})`}</option>`).join('');
      picker.dataset.signature = signature;
    }
    if (!choices.includes(selected)) selected = enabled[0] || choices[0];
    picker.value = selected;
    $('#sp-camera-label').textContent = sensorName(selected);
    $('#sp-channel-label').textContent = selected === 'lidar' ? 'LiDAR' : t('RGB 프리뷰');
    $('#sp-pose-label').textContent = `${sensorName(selected)} · ${t('차량 기준 좌표')}`;
    const p = pose(), active = enabled.includes(selected);
    for (const key of ['x','y','z','yaw','pitch','roll']) {
      const input = $('#sp-'+key);
      if (document.activeElement !== input) input.value = p[key];
      input.disabled = !active;
    }
    renderMap('top', draft); renderMap('side', draft);
    if (!loaded.sensor) {
      const desc = $('[data-sp-stage="sensor"] .sp-no-frame p');
      if (desc) desc.textContent = !active ? t('이 센서를 활성화한 뒤 프리뷰에 적용하세요.') : t('CARLA를 연결하고 프리뷰를 시작하세요.');
    }
  }
  function renderMap(kind, draft) {
    const svg = $(`[data-sp-map="${kind}"]`); if (!svg) return;
    const entries = Object.keys(draft.sensors || {}).map(id=>[id,pose(id)]);
    let scale, ox = 210, oy, project, inverse;
    if (kind === 'top') {
      const extentX = Math.max(3.7, ...entries.map(([,p])=>Math.abs(p.x)+.7));
      const extentY = Math.max(4, ...entries.map(([,p])=>Math.abs(p.y)+.7));
      scale = Math.min(180/extentY,85/extentX);oy=112;
      project = p => [ox+p.y*scale, oy-p.x*scale];inverse = (x,y)=>({x:(oy-y)/scale,y:(x-ox)/scale});
    } else {
      const extentX = Math.max(4.5,...entries.map(([,p])=>Math.abs(p.x)+.7));
      const extentZ = Math.max(3.9,...entries.map(([,p])=>p.z+.7));
      scale=Math.min(182/extentX,158/extentZ);oy=193;
      project = p=>[ox+p.x*scale,oy-p.z*scale];inverse=(x,y)=>({x:(x-ox)/scale,z:(oy-y)/scale});
    }
    if(drag?.kind===kind){
      scale=drag.scale;ox=drag.ox;oy=drag.oy;
      project=kind==='top'?(p=>[ox+p.y*scale,oy-p.x*scale]):(p=>[ox+p.x*scale,oy-p.z*scale]);
      inverse=drag.inverse;
    }
    mapTransforms[kind] = {inverse,scale,ox,oy};
    const line = (x1,y1,x2,y2,css='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${css}"/>`;
    let grid='';for(let i=-10;i<=10;i++){
      const gx=ox+i*scale;if(gx>15&&gx<405)grid+=line(gx,20,gx,201,'sp-grid-line');
      const gy=oy-i*scale;if(gy>20&&gy<201)grid+=line(15,gy,405,gy,'sp-grid-line');
    }
    const car=kind==='top'?`<rect x="${ox-.95*scale}" y="${oy-2.2*scale}" width="${1.9*scale}" height="${4.4*scale}" rx="${.35*scale}" class="sp-car"/><path d="M ${ox-.73*scale} ${oy-.9*scale} L ${ox+.73*scale} ${oy-.9*scale} L ${ox+.6*scale} ${oy-1.4*scale} L ${ox-.6*scale} ${oy-1.4*scale} Z" class="sp-window"/>`:`<path d="M ${ox-2.2*scale} ${oy-.4*scale} L ${ox-2.2*scale} ${oy-.9*scale} L ${ox-1.35*scale} ${oy-1.05*scale} L ${ox-.85*scale} ${oy-1.5*scale} L ${ox+.8*scale} ${oy-1.5*scale} L ${ox+1.35*scale} ${oy-1.05*scale} L ${ox+2.2*scale} ${oy-.8*scale} L ${ox+2.2*scale} ${oy-.4*scale} Z" class="sp-car"/><circle cx="${ox-1.35*scale}" cy="${oy-.35*scale}" r="${.34*scale}" class="sp-wheel"/><circle cx="${ox+1.35*scale}" cy="${oy-.35*scale}" r="${.34*scale}" class="sp-wheel"/>`;
    const markers=entries.map(([id,p])=>{
      const [x,y]=project(p), chosen=id===selected;
      const angle=kind==='top'?(p.yaw-90)*Math.PI/180:-p.pitch*Math.PI/180;
      const config=draft.sensors[id],horizontal=Number(config?.fov)||100;
      const fov=kind==='top'?horizontal:2*Math.atan(Math.tan(horizontal*Math.PI/360)*(Number(config?.height)||300)/(Number(config?.width)||400))*180/Math.PI;
      const half=Math.max(1,Math.min(160,fov))*Math.PI/360,radius=44;
      const wedge=chosen&&id!=='lidar'?`<path d="M ${x} ${y} L ${x+Math.cos(angle-half)*radius} ${y+Math.sin(angle-half)*radius} A ${radius} ${radius} 0 0 1 ${x+Math.cos(angle+half)*radius} ${y+Math.sin(angle+half)*radius} Z" class="sp-fov"><title>FOV ${fmt(fov)}°</title></path>`:'';
      return `<g data-sp-sensor="${esc(id)}" class="sp-mount ${chosen?'selected':''}"><title>${esc(sensorName(id))}: X ${fmt(p.x)}, Y ${fmt(p.y)}, Z ${fmt(p.z)} m</title>${wedge}${chosen?line(x,y,x+Math.cos(angle)*23,y+Math.sin(angle)*23,'sp-direction'):''}<circle cx="${x}" cy="${y}" r="${chosen?7:5}"/>${chosen?`<text x="${x>300?x-12:x+12}" y="${Math.max(23,y-11)}" text-anchor="${x>300?'end':'start'}">${esc(sensorName(id))}</text>`:''}</g>`;
    }).join('');
    svg.innerHTML=`${grid}${line(15,oy,405,oy,'sp-axis')}${line(ox,20,ox,201,'sp-axis')}${car}<circle cx="${ox}" cy="${oy}" r="2.5" class="sp-origin"/>${markers}<text x="${kind==='top'?ox:393}" y="${kind==='top'?16:oy-8}" text-anchor="${kind==='top'?'middle':'end'}" class="sp-axis-label">+X · ${t('앞')}</text><text x="399" y="${kind==='top'?oy-8:17}" text-anchor="end" class="sp-axis-label">${kind==='top'?'+Y · '+t('오른쪽'):'+Z'}</text><text x="18" y="214" class="sp-axis-label">${t('저장 전 배치도')} · 1 m/grid</text>`;
    svg.classList.toggle('sp-map-disabled',!draft.sensors?.[selected]);
  }
  function active() {return preview.status==='Running' || preview.busy===true;}
  function blockers() {
    const missing=[];
    if(state.online===false)missing.push(t('서버에 연결할 수 없습니다. 연결 상태를 확인하세요.'));
    if((state.jobs||[]).some(j=>j.status==='Running'))missing.push(t('연구 작업이 실행 중입니다. 작업이 끝난 뒤 프리뷰를 시작하세요.'));
    if(state.carla_local?.task?.status==='Running')missing.push(t('CARLA 설치·기동이 진행 중입니다. 준비가 끝날 때까지 기다리세요.'));
    if(!state.carla?.connected&&!active())missing.push(t('CARLA 서버를 먼저 연결하세요.'));
    try {const d=readDraft();if(!Object.keys(d.sensors||{}).some(id=>id!=='lidar'))missing.push(t('활성 카메라를 하나 이상 선택하세요.'));}catch(error){missing.push(error.message);}
    return missing;
  }
  function blocker() {return blockers()[0]||'';}
  function selectedMetadata() {return (preview.channels||[]).find(c=>c.id===(selected==='lidar'?'lidar':selected+'_rgb'));}
  function requestedFPS() {
    const value=Number(state.settings?.preview_fps??24);
    return Number.isInteger(value)&&value>=1&&value<=60?value:24;
  }
  function sensorFPS() {
    const value=Number(preview.preview_fps??selectedMetadata()?.requested_fps??preview.requested_fps);
    return Number.isInteger(value)&&value>=1&&value<=60?value:requestedFPS();
  }
  function unappliedFPS() {return active()&&requestedFPS()!==Number(preview.requested_fps??preview.preview_fps??requestedFPS());}
  function refreshFPS() {
    const metadata=selectedMetadata();
    let limit=Math.min(requestedFPS(),sensorFPS());
    // Observed output limits repeat downloads; status polling can raise this again.
    const measured=metadata?.measured_fps;
    if(finite(measured))limit=Math.min(limit,Math.max(1,Math.ceil(measured)));
    return Math.max(1,limit);
  }
  function renderStatus() {
    if(!root?.isConnected)return;
    const running=active(),ready=preview.ready===true,canApply=running&&preview.phase==='live',missing=blockers(),why=missing[0]||'';
    const badge=$('#sp-status');badge.className='badge '+(preview.status||'Idle');badge.textContent=t(phaseNames[preview.phase]||statusNames[preview.status]||'대기');
    const start=$('[data-sp="start"]'),apply=$('[data-sp="apply"]'),stop=$('[data-sp="stop"]');
    start.disabled=Boolean(why)||running||requestPending;start.title=why||(running?t('프리뷰가 이미 실행 중입니다.'):'');
    apply.disabled=Boolean(why)||!canApply||requestPending||applyPending;apply.title=why||(!running?t('프리뷰를 시작한 뒤 설정을 적용하세요.'):!canApply?t('프리뷰 준비·정리 중에는 설정을 적용할 수 없습니다.'):'');
    stop.disabled=!running||preview.stop_requested||requestPending;stop.title=preview.stop_requested?t('중지 요청 중'):'';
    const target=$('#sp-blocker');
    if(!running)missing.push(t('실제 센서를 확인하려면 프리뷰를 시작하세요.'));
    if(dirty)missing.push(t('변경한 설정을 프리뷰에 적용하세요.'));
    if(unappliedFPS())missing.push(t('저장한 FPS를 프리뷰에 적용하세요.'));
    if(running&&!ready)missing.push(t(preview.phase==='live'?'선택 센서의 실제 프레임 수신을 기다리세요.':'센서 준비가 끝날 때까지 기다리세요.'));
    if(!safeDraft().sensors?.[selected])missing.push(t('선택 센서를 활성화하고 프리뷰에 적용하세요.'));
    else if(running&&ready&&!selectedMetadata()?.ready)missing.push(t('선택 센서의 실제 프레임 수신을 기다리세요.'));
    target.className='sp-feedback sp-requirements';
    target.innerHTML=missing.length?`<strong>${t('다음 단계의 준비 조건')}</strong><ul>${[...new Set(missing)].map(item=>`<li>${esc(diagnostic(item))}</li>`).join('')}</ul>${!state.carla?.connected&&!running?`<a href="#simulation" data-action="open-local-carla" class="text-link">${t('CARLA 설치·연결로 이동')}</a>`:''}`:'';
    target.hidden=!missing.length;
    const notice=$('#sp-notice');
    if(noticeError||statusError||preview.error){notice.textContent=diagnostic(noticeError||statusError||preview.error);notice.className='sp-feedback sp-error';}
    else if(dirty){notice.textContent=t('프리뷰에 적용하지 않은 변경이 있습니다.');notice.className='sp-feedback sp-pending';}
    else if(unappliedFPS()){notice.textContent=t('저장한 FPS를 프리뷰에 적용하세요.');notice.className='sp-feedback sp-pending';}
    else if(ready&&running){notice.textContent=t('프리뷰에 현재 설정이 적용되었습니다.');notice.className='sp-feedback sp-success';}
    else {notice.textContent=!preview.phase||preview.phase==='idle'?t('CARLA를 연결하고 프리뷰를 시작하세요.'):diagnostic(preview.message||'');notice.className='sp-feedback';}
    notice.hidden=!notice.textContent;
    const rate=$('#sp-rate'),metadata=selectedMetadata();
    rate.textContent=`${t('설정한 목표 FPS')} ${requestedFPS()} · ${t('현재 센서 요청 FPS')} ${active()?sensorFPS():'—'} · ${t('실제 수신 FPS')} ${finite(metadata?.received_fps)?fmt(metadata.received_fps):'—'} · ${t('실제 출력 FPS')} ${finite(metadata?.measured_fps)?fmt(metadata.measured_fps):'—'} · ${t('화면 갱신 상한')} ${fmt(refreshFPS())} FPS. ${t('실제 FPS는 CARLA 실행 속도에 따라 낮아질 수 있습니다.')}`;
    for(const slot of ['sensor']) {
      const frame=loaded[slot];if(!frame)continue;
      const raw=frame.received_at,ms=typeof raw==='number'&&raw<1e11?raw*1000:Date.parse(raw);
      const stale=Number.isFinite(ms)&&Date.now()-ms>5000;
      const fresh=$(`[data-sp-fresh="${slot}"]`);
      fresh.textContent=!running?t('마지막 수신 프레임'):stale?t('최신 프레임 대기'):'';
      fresh.classList.toggle('sp-stale',stale||!running);
    }
  }
  function acceptStatus(value) {
    if(!value||typeof value!=='object')return;
    // A delayed SSE snapshot must not roll back a newer apply response.
    if(value.session_id&&value.session_id===preview.session_id&&Number(value.requested_revision??0)<Number(preview.requested_revision??0))return;
    if(preview.session_id&&value.session_id!==preview.session_id)clearFrame('sensor');
    preview=value;lastStatus=Date.now();
    for(const [slot,channel] of [['sensor',selected==='lidar'?'lidar':selected+'_rgb']]){
      const metadata=(preview.channels||[]).find(c=>c.id===channel);
      if(metadata?.ready===false||metadata&&metadata.frame===null||preview.phase==='applying'||preview.requested_revision>preview.revision)clearFrame(slot);
    }
    if(applyPending&&preview.revision>=preview.requested_revision&&preview.phase==='live')applyPending=false;
    renderStatus();
    scheduleFrame();
  }
  function update(nextState) {
    state=nextState||state;
    if(!root?.isConnected){if(root)destroy();return;}
    if(state.sensor_preview)acceptStatus(state.sensor_preview);
    renderStatus();
  }
  async function requestStatus() {
    if(statusPending||!root?.isConnected)return;
    statusPending=true;const run=generation;
    try {const result=await hooks.api('/api/sensor-preview/status');if(run===generation){statusError='';acceptStatus(result);}}
    catch(error){statusError=error.message||t('프리뷰 상태를 불러올 수 없습니다.');const notice=$('#sp-notice');if(notice){notice.textContent=diagnostic(statusError);notice.hidden=false;notice.className='sp-feedback sp-error';}}
    finally{statusPending=false;}
  }
  function clearFrame(slot) {
    delete loaded[slot];cancelImage(slot);
    const stage=$(`[data-sp-stage="${slot}"]`);if(!stage)return;
    stage.querySelector('img')?.remove();stage.querySelector('.sp-no-frame').hidden=false;
    $(`[data-sp-frame="${slot}"]`).textContent=t('프레임')+' —';$(`[data-sp-time="${slot}"]`).textContent=t('촬영 시각')+' —';$(`[data-sp-fresh="${slot}"]`).textContent='';
  }
  function cancelImage(slot) {
    imageTokens[slot]=(imageTokens[slot]||0)+1;pending[slot]=false;
    const request=imageRequests[slot];delete imageRequests[slot];
    if(request){clearTimeout(request.timeout);request.controller.abort();request.img.onload=null;request.img.onerror=null;request.img.removeAttribute('src');}
  }
  function scheduleFrame(delay=0) {
    if(frameTimer||pending.sensor||!root?.isConnected||document.hidden||!active())return;
    const meta=selectedMetadata();
    if(!meta||meta.ready===false||meta.frame===null||preview.phase==='applying'||preview.requested_revision>preview.revision)return;
    const wait=Math.ceil(Math.max(delay,1000/refreshFPS()-(Date.now()-lastImageRequest),0));
    frameTimer=setTimeout(()=>{frameTimer=null;frame('sensor',selected==='lidar'?'lidar':selected+'_rgb');},wait);
  }
  async function frame(slot,channel) {
    const meta=(preview.channels||[]).find(c=>c.id===channel);
    if(!meta||meta.ready===false||meta.frame===null||preview.phase==='applying'||preview.requested_revision>preview.revision||pending[slot]||!root?.isConnected||document.hidden||!active())return;
    // The latest JPEG can change between low-frequency status snapshots.
    // Its revision and session stay guarded; a status frame number is not a cache key.
    const run=generation,selectedWhenRequested=selected,sessionWhenRequested=preview.session_id,revisionWhenRequested=preview.revision;
    pending[slot]=true;const ticket=imageTokens[slot]=(imageTokens[slot]||0)+1;
    lastImageRequest=Date.now();
    const img=new Image(),controller=new AbortController();img.alt=sensorName(selected)+' · '+(selected==='lidar'?'LiDAR':'RGB');img.decoding='async';
    const finish=(delay=0)=>{
      if(ticket!==imageTokens[slot])return;
      clearTimeout(imageRequests[slot]?.timeout);delete imageRequests[slot];pending[slot]=false;scheduleFrame(delay);
    };
    img.onload=()=>{
      if(ticket!==imageTokens[slot])return;
      if(run!==generation||!root?.isConnected||document.hidden||preview.session_id!==sessionWhenRequested||preview.revision!==revisionWhenRequested||preview.requested_revision>preview.revision||preview.phase==='applying'||selected!==selectedWhenRequested){finish();return;}
      const stage=$(`[data-sp-stage="${slot}"]`);stage.querySelector('img')?.remove();stage.querySelector('.sp-no-frame').hidden=true;stage.appendChild(img);
      const reported=selectedMetadata()||meta;
      if(loaded[slot]?.frame!==reported.frame||loaded[slot]?.received_at!==reported.received_at){
        $(`[data-sp-frame="${slot}"]`).textContent=`${t('마지막 상태 확인 기준')} · ${t('프레임')} ${reported.frame??'—'} · ${img.naturalWidth} × ${img.naturalHeight}`;
        $(`[data-sp-time="${slot}"]`).textContent=`${t('촬영 시각')} ${stamp(reported.received_at)}${finite(reported.timestamp)?` · ${t('시뮬레이션 시각')} ${reported.timestamp.toFixed(2)} s`:''}`;
      }
      loaded[slot]={...reported};finish();
    };
    img.onerror=()=>{if(ticket!==imageTokens[slot])return;if(run===generation&&preview.revision===revisionWhenRequested&&preview.session_id===sessionWhenRequested){const fresh=$(`[data-sp-fresh="${slot}"]`);if(fresh)fresh.textContent=t('새 프레임을 불러오지 못했습니다.');}finish(500);};
    imageRequests[slot]={img,controller,timeout:setTimeout(()=>{
      if(ticket!==imageTokens[slot])return;
      const fresh=$(`[data-sp-fresh="${slot}"]`);if(fresh)fresh.textContent=t('새 프레임을 불러오지 못했습니다.');
      cancelImage(slot);scheduleFrame(500);
    },10000)};
    // The endpoint sends no-store; a stable URL avoids a new cache entry per poll.
    const url=`/api/sensor-preview/frame?channel=${encodeURIComponent(channel)}&session_id=${encodeURIComponent(preview.session_id||'')}&revision=${encodeURIComponent(preview.revision||0)}`;
    try {if(hooks.imageSource)await hooks.imageSource(img,url,{signal:controller.signal});else img.src=url;}catch{img.onerror?.();}
  }
  async function tick() {
    if(!root?.isConnected){destroy();return;}
    if(document.hidden)return;
    if(Date.now()-lastStatus>2800)requestStatus();
    scheduleFrame();renderStatus();
  }
  async function command(kind) {
    if(requestPending)return;
    if(kind==='edit'){
      const field=form?.elements.namedItem(selected+'.x')||form?.elements.namedItem('enable_'+selected);
      if(field){const details=field.closest('details');if(details)details.open=true;field.scrollIntoView({block:'center',behavior:'smooth'});field.focus({preventScroll:true});}return;
    }
    if(kind==='logs'){
      const details=$('.sp-logs');details.open=true;$('#sp-logs').textContent=t('로그를 불러오는 중입니다.');
      try{const result=await hooks.api('/api/sensor-preview/logs');$('#sp-logs').textContent=result.text||t('기록된 로그가 없습니다.');}catch(error){$('#sp-logs').textContent=error.message;}return;
    }
    const stopping=kind==='stop';
    try {
      if(!stopping&&blocker())throw new Error(blocker());
      if(kind==='start'&&hooks.confirm&&!await hooks.confirm(t('실제 센서 프리뷰를 시작할까요?'),t('연결된 CARLA에 프리뷰용 차량과 센서를 생성합니다. 수집 데이터셋에는 저장하지 않습니다.'),t('프리뷰 시작')))return;
      if(stopping&&hooks.confirm&&!await hooks.confirm(t('프리뷰를 중지할까요?'),t('프리뷰 차량과 센서를 정리합니다. 저장한 센서 구성은 유지됩니다.'),t('프리뷰 중지')))return;
      requestPending=true;noticeError='';renderStatus();
      const payload=stopping?undefined:{sensors:readDraft()};
      const result=await hooks.api(stopping?'/api/sensor-preview/stop':kind==='start'?'/api/sensor-preview/start':'/api/sensor-preview',kind==='apply'?'PUT':'POST',payload);
      if(!stopping){dirty=false;applyPending=kind==='apply';}
      acceptStatus(result);await requestStatus();tick();
    }catch(error){noticeError=error.message;const notice=$('#sp-notice');if(notice){notice.textContent=diagnostic(error.message);notice.hidden=false;notice.className='sp-feedback sp-error';}}
    finally{requestPending=false;renderStatus();}
  }
  function movePointer(event) {
    if(!drag||!root?.isConnected)return;
    const svg=drag.svg,point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
    const matrix=svg.getScreenCTM();if(!matrix)return;
    const local=point.matrixTransform(matrix.inverse()),values=mapTransforms[drag.kind].inverse(local.x,local.y);
    for(const key of Object.keys(values))values[key]=Math.max(key==='z'?.1:-10,Math.min(10,values[key]));
    writePose(values);
  }
  function mount(options={}) {
    const next=document.querySelector('#sensor-preview-control');
    if(next===root&&root?.isConnected){hooks={...hooks,...options};update(options.state);return;}
    destroy();if(!next)return;root=next;hooks=options;state=options.state||{};preview=state.sensor_preview||{};form=document.querySelector('#sensor-form');
    if(typeof hooks.api!=='function')throw new Error('SensorPreview.mount requires api(path,method,body)');
    root.addEventListener('click',event=>{const button=event.target.closest('[data-sp]');if(button&&!button.disabled)command(button.dataset.sp);});
    $('#sp-sensor').addEventListener('change',event=>{selected=event.target.value;generation++;pending.sensor=false;clearFrame('sensor');renderDraft();tick();});
    root.querySelectorAll('[data-sp-pose]').forEach(input=>input.addEventListener('input',()=>{if(input.value.trim()!==''&&Number.isFinite(Number(input.value)))writePose({[input.dataset.spPose]:Number(input.value)});}));
    root.querySelectorAll('[data-sp-map]').forEach(svg=>{
      svg.addEventListener('pointerdown',event=>{if(event.button!==0)return;const marker=event.target.closest('[data-sp-sensor]');if(marker){selected=marker.dataset.spSensor;clearFrame('sensor');renderDraft();}if(!safeDraft().sensors?.[selected])return;event.preventDefault();svg.focus();drag={svg,kind:svg.dataset.spMap,pointer:event.pointerId,...mapTransforms[svg.dataset.spMap]};svg.setPointerCapture(event.pointerId);movePointer(event);});
      svg.addEventListener('pointermove',event=>{if(drag?.pointer===event.pointerId)movePointer(event);});
      const release=event=>{if(drag?.pointer===event.pointerId){if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);drag=null;renderDraft();}};
      svg.addEventListener('pointerup',release);svg.addEventListener('pointercancel',release);
      svg.addEventListener('keydown',event=>{if(!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(event.key)||!safeDraft().sensors?.[selected])return;event.preventDefault();const delta=event.shiftKey?.5:.1,p=pose(),kind=svg.dataset.spMap;const key=kind==='top'?(event.key==='ArrowUp'||event.key==='ArrowDown'?'x':'y'):(event.key==='ArrowUp'||event.key==='ArrowDown'?'z':'x');const direction=event.key==='ArrowUp'||event.key==='ArrowRight'?1:-1;writePose({[key]:Math.max(key==='z'?.1:-10,Math.min(10,p[key]+delta*direction))});});
    });
    formListener=()=>{dirty=true;noticeError='';if(!renderScheduled){renderScheduled=true;requestAnimationFrame(()=>{renderScheduled=false;renderDraft();renderStatus();});}};
    form?.addEventListener('input',formListener);form?.addEventListener('change',formListener);
    visibilityListener=()=>{if(document.hidden){clearTimeout(frameTimer);frameTimer=null;cancelImage('sensor');}else{requestStatus();scheduleFrame();}};
    document.addEventListener('visibilitychange',visibilityListener);
    renderDraft();acceptStatus(preview);requestStatus();timer=setInterval(tick,1000);
  }
  function destroy() {
    generation++;if(timer)clearInterval(timer);timer=null;clearTimeout(frameTimer);frameTimer=null;cancelImage('sensor');lastImageRequest=0;
    if(visibilityListener)document.removeEventListener('visibilitychange',visibilityListener);visibilityListener=null;
    if(form&&formListener){form.removeEventListener('input',formListener);form.removeEventListener('change',formListener);}
    form=null;formListener=null;root=null;drag=null;statusPending=false;requestPending=false;applyPending=false;
    for(const key of Object.keys(loaded))delete loaded[key];for(const key of Object.keys(pending))delete pending[key];
  }
  window.SensorPreview={html,mount,update,destroy,readDraft};
})();
