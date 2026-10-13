"""A resizable Pygame workbench around the existing SDV experiment screens."""
from pathlib import Path
import time
import pygame
from sdv.runtime.processes import ROOT, OWNED

INK=(29,47,68)
BLUE=(34,91,151)
PAPER=(248,250,253)
LINE=(208,221,233)
TEAL=(18,112,111)
RED=(175,55,65)
FONT=next((p for p in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if Path(p).is_file()),None)
STAGES={-1:0,-2:0,0:0,1:0,2:0,3:1,4:2,5:2,6:2,7:2,8:2,9:3,10:3,11:2,12:3}
HELP={-1:'센서를 설정하고 데이터를 수집한 뒤, 모델을 학습하고 결과를 확인합니다.',
      -2:'기본 배포만 설치합니다. 중단한 다운로드는 다음 시도에서 이어받습니다.',
      0:'타운과 카메라를 선택하세요. 처음에는 Town01 · 전면 카메라 · 3장으로 시작하세요.',
      1:'이 화면은 저장소에 포함된 예시입니다. 센서 위치 확인을 누르면 실제 CARLA에 연결합니다.',
      2:'차량 이미지를 클릭해 위치를 바꾸거나 숫자를 입력한 뒤 Enter로 적용하세요.',
      3:'실제 생성된 데이터가 도착하면 미리보기가 표시됩니다. 중지해도 저장된 파일은 유지됩니다.',
      4:'에포크·배치·학습률을 입력하세요. 설정 저장과 실제 학습 준비 상태를 구분합니다.',
      5:'센서 설정을 바탕으로 사용할 입력과 출력, 모델 구성 요소를 선택하세요.',
      6:'모델 설정을 확인하고 대기열에 추가하세요.',
      7:'실험 설정을 대기열에 저장합니다. 실제 학습 전 데이터·텐서·실행 조건을 검사합니다.',
      8:'실행 상태와 로그를 확인하세요. 중지는 이번 SDV에서 시작한 작업에만 적용됩니다.',
      9:'완료된 실험을 선택하세요. 결과가 없으면 먼저 데이터와 학습 준비 상태를 확인하세요.',
      10:'저장된 그래프·이미지·지표를 확인하세요.',11:'결과를 바탕으로 새 실험 설정을 준비하세요.'}

class Shell:
    def __init__(self):
        from sdv.ui.responsive import Renderer
        desktop=pygame.display.get_desktop_sizes()[0]
        available=(max(640,desktop[0]-48),max(480,desktop[1]-64))
        self.minimum=(min(1024,available[0]),min(700,available[1]))
        size=(min(1280,available[0]),min(820,available[1]))
        self.display=pygame.display.set_mode(size,pygame.RESIZABLE)
        pygame.display.set_caption('SDV | 자율주행 실험 작업실')
        self.renderer=Renderer(FONT);self.context={};self.current_state=-1
        self.notice='설정 준비 완료. CARLA는 센서 미리보기에서 시작합니다.'
        self.level='info';self.focus=-1;self.focus_state=None;self.controls=[]
        self.scale=1;self.origin=(0,0);self.details=False;self.details_scroll=0
        self.busy=False;self.pending_since=0;self.user_scale=1.0;self.capture=None
        self.chrome_home=pygame.Rect(0,0,0,0);self.chrome_stop=pygame.Rect(0,0,0,0)
        self.status_rect=pygame.Rect(0,0,0,0);self.details_rect=pygame.Rect(0,0,0,0)
    def bind_context(self,context):self.context=context
    def resize(self,size):
        self.display=pygame.display.set_mode((max(self.minimum[0],size[0]),max(self.minimum[1],size[1])),pygame.RESIZABLE)
    def pixel_ratio(self):
        window=pygame.display.get_window_size();surface=self.display.get_size()
        return surface[0]/max(1,window[0]),surface[1]/max(1,window[1])
    def to_display(self,pos):
        x,y=self.pixel_ratio();return (pos[0]*x,pos[1]*y)
    def event_position(self,pixel):
        x,y=self.pixel_ratio();return (round(pixel[0]/x),round(pixel[1]/y))
    def tell(self,message,level='info'):
        self.notice=str(message);self.level=level;self.details_scroll=0
    def to_canvas(self,pos):return self.renderer.logical(pos)
    def map_pointer(self,event):
        pos=self.to_display(getattr(event,'pos',pygame.mouse.get_pos()))
        control=self.capture or self.renderer.target_at(pos)
        if event.type==pygame.MOUSEBUTTONDOWN and control:
            self.focus=self.renderer.controls.index(control)
            if control.affine:self.capture=control
        point=self.renderer.logical(pos,control)
        if event.type==pygame.MOUSEBUTTONUP:self.capture=None
        return point
    def control_position(self,logical):
        rect=pygame.Rect(logical)
        for item in self.renderer.controls:
            if item.target==rect or item.target.collidepoint(rect.center):
                if item.body:
                    view=self.renderer.m.content;delta=0
                    if item.rect.top<view.top:delta=item.rect.top-view.top-12
                    elif item.rect.bottom>view.bottom:delta=item.rect.bottom-view.bottom+12
                    if delta:
                        self.renderer.scroll=max(0,min(self.renderer.max_scroll,self.renderer.scroll+delta))
                        self._paint(None,self.current_state)
                        return self.control_position(logical)
                return self.event_position(item.rect.center)
        raise LookupError('현재 화면에서 해당 제어를 찾을 수 없습니다: '+str(tuple(rect)))
    def prepare_controls(self,state,controls):
        if state!=self.focus_state:self.focus_state=state;self.focus=-1
    def keyboard(self,event):
        if event.type!=pygame.KEYDOWN:return None
        if event.key==pygame.K_TAB and self.renderer.controls:
            self.focus=(self.focus+(-1 if event.mod & pygame.KMOD_SHIFT else 1))%len(self.renderer.controls)
            item=self.renderer.controls[self.focus]
            if item.body:
                view=self.renderer.m.content
                if item.rect.top<view.top:self.renderer.scroll=max(0,self.renderer.scroll+item.rect.top-view.top-8)
                elif item.rect.bottom>view.bottom:self.renderer.scroll=min(self.renderer.max_scroll,self.renderer.scroll+item.rect.bottom-view.bottom+8)
            return 'handled'
        if event.key in (pygame.K_RETURN,pygame.K_SPACE) and 0<=self.focus<len(self.renderer.controls):
            return pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=self.renderer.controls[self.focus].target.center,logical=True)
        return None
    def handle_ui_event(self,event):
        if event.type==pygame.MOUSEWHEEL or (event.type==pygame.MOUSEBUTTONDOWN and event.button in (4,5)):
            delta=event.y if event.type==pygame.MOUSEWHEEL else (1 if event.button==4 else -1)
            if self.details:self.details_scroll=max(0,self.details_scroll-delta*48)
            else:self.renderer.scroll=max(0,min(self.renderer.max_scroll,self.renderer.scroll-delta*64))
            return True
        if event.type==pygame.KEYDOWN:
            if self.details and event.key==pygame.K_ESCAPE:self.details=False;return True
            if event.mod & pygame.KMOD_CTRL and event.key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_MINUS,pygame.K_0):
                self.user_scale=1 if event.key==pygame.K_0 else max(.9,min(1.5,self.user_scale+(.1 if event.key!=pygame.K_MINUS else -.1)))
                return True
            if event.key==pygame.K_TAB and (self.context.get('active_input') or self.context.get('hyperparameter_input') or self.context.get('train_modal_select_input')):
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_RETURN,unicode='\r',mod=0))
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_TAB,unicode='\t',mod=event.mod));return True
        if self.details and event.type in (pygame.MOUSEBUTTONDOWN,pygame.MOUSEBUTTONUP,pygame.MOUSEMOTION):
            pos=self.to_display(getattr(event,'pos',pygame.mouse.get_pos()))
            if self.details_rect.collidepoint(pos):return True
        return False
    def present(self,canvas,state):
        self._paint(canvas,state)
    def _paint(self,canvas,state):
        self.current_state=state;self.display.fill(PAPER)
        ratio=self.pixel_ratio();density=self.user_scale*max(ratio)
        self.renderer.render(self.display,self.context,state,self,density)
        r=self.renderer;m=r.m;p=m.px;w,h=self.display.get_size()
        self.display.set_clip(None)
        pygame.draw.rect(self.display,PAPER,(0,0,w,m.header))
        r.text((m.margin,p(16),p(92),p(42)),'SDV','section',BLUE)
        left=m.margin+p(140);right=w-m.margin-p(180);slot=max(p(75),(right-left)//4)
        for i,label in enumerate(['설정','데이터 수집','모델 학습','결과 확인']):
            box=pygame.Rect(left+i*slot,p(17),slot-p(8),p(40))
            if i==STAGES.get(state,0):pygame.draw.rect(self.display,(226,236,248),box,border_radius=5)
            r.text(box.inflate(-p(12),0),f'{i+1} {label}','label',BLUE if i==STAGES.get(state,0) else (87,108,128),center=True)
        self.chrome_home=pygame.Rect(w-m.margin-p(160),p(17),p(72),p(40));self.chrome_stop=pygame.Rect(w-m.margin-p(80),p(17),p(80),p(40))
        for rect,label,color in [(self.chrome_home,'홈',BLUE),(self.chrome_stop,'중지',RED)]:
            pygame.draw.rect(self.display,LINE,rect,1,border_radius=5);r.text(rect,label,'label',color,center=True)
        r.text((m.margin,p(66),w-2*m.margin,p(28)),HELP.get(state,'실험 상태와 설정을 확인하세요.'),'label',(87,108,128))
        self.status_rect=pygame.Rect(0,h-m.footer,w,m.footer);pygame.draw.rect(self.display,PAPER,self.status_rect);pygame.draw.line(self.display,LINE,self.status_rect.topleft,self.status_rect.topright)
        active=[j for j in OWNED if j.poll() is None and not j.stopped]
        message=(f'{active[0].label} {int(time.monotonic()-active[0].started)}초 / 90초 · ' if active else '')+self.notice
        if self.busy:message='CARLA 연결 준비 중… '+message
        r.text((m.margin,h-m.footer+p(8),w-2*m.margin-p(164),m.footer-p(16)),message,'label',RED if self.level=='error' else TEAL)
        r.text((w-m.margin-p(148),h-m.footer+p(8),p(148),m.footer-p(16)),'상태 자세히 보기','small',BLUE,center=True)
        if self.details:
            panel=pygame.Rect(m.margin,max(p(100),h-m.footer-p(360)),w-2*m.margin,min(p(340),h-m.footer-p(100)))
            self.details_rect=panel;r.box(panel)
            r.text((panel.x+p(20),panel.y+p(16),panel.w-p(40),p(32)),'상태 상세 · 상태 표시줄 또는 Esc로 닫기','card')
            viewport=pygame.Rect(panel.x+p(20),panel.y+p(66),panel.w-p(40),panel.h-p(88));lines=r.wrap(message,viewport.w,'body');line_h=r.font('body').get_linesize()+p(6)
            self.details_scroll=min(self.details_scroll,max(0,len(lines)*line_h-viewport.h))
            self.display.set_clip(viewport)
            for i,line in enumerate(lines):r.text((viewport.x,viewport.y+i*line_h-self.details_scroll,viewport.w,line_h),line,'body',RED if self.level=='error' else INK)
            self.display.set_clip(None)
        elif r.tooltips:
            mouse=self.to_display(pygame.mouse.get_pos())
            for rect,text in reversed(r.tooltips):
                if rect.collidepoint(mouse):
                    width=min(w-2*m.margin,p(620));parts=r.wrap(text,width-p(24),'label')[:6];height=p(20)+len(parts)*r.font('label').get_linesize();top=min(h-m.footer-height,mouse[1]+p(20));box=pygame.Rect(min(w-m.margin-width,max(m.margin,mouse[0])),max(m.header,top),width,height)
                    r.box(box);r.text(box.inflate(-p(24),-p(12)),text,'label',INK,len(parts));break
        pygame.display.flip()
    def draw_menu(self,canvas,buttons,assets):
        # Keep the workflow's logical menu targets; the native renderer paints it.
        canvas.fill(PAPER)
