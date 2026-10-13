"""Native Pygame layout and shared control geometry for the existing workflow."""
import math
import time
from dataclasses import dataclass
import pygame
from sdv.paths import ROOT
from sdv.runtime.processes import OWNED

INK=(29,47,68); BLUE=(34,91,151); PAPER=(248,250,253); LINE=(208,221,233)
TEAL=(18,112,111); RED=(175,55,65); MUTED=(87,108,128); WHITE=(255,255,255)
LABELS={'front_cam':'전면 카메라','front_cam_pv':'전면 원근','left_cam':'왼쪽 카메라','right_cam':'오른쪽 카메라','rear_cam':'후면 카메라','rear_left_cam':'후면 왼쪽','rear_right_cam':'후면 오른쪽','lidar':'라이다','Best_Loss':'최저 손실','Best_iou':'최고 IoU','Last_Epoch':'마지막 에포크','Seg':'분할','Depth':'깊이','od':'객체 검출'}
TITLES={-1:'실험을 시작할 준비',-2:'CARLA 설치',0:'타운과 센서 설정',1:'수집 설정 검토',2:'센서 위치와 카메라',3:'데이터 수집',4:'학습 조건 설정',5:'모델 구성',6:'모델 설정 검토',7:'학습 대기열',8:'학습 실행 상태',9:'저장된 결과',10:'결과 자세히 보기',11:'재학습 설정'}

@dataclass
class Metrics:
    width:int
    height:int
    density:float=1
    def __post_init__(self):
        self.density=max(.9,min(1.75,self.density))
        self.margin=self.px(24);self.gap=self.px(20);self.row=self.px(44)
        self.header=self.px(100);self.action=self.px(68);self.footer=self.px(48)
        wide=min(self.width-2*self.margin,self.px(1600))
        self.content=pygame.Rect((self.width-wide)//2,self.header,wide,max(self.px(100),self.height-self.header-self.action-self.footer))
    def px(self,value):return max(1,round(value*self.density))

@dataclass
class Control:
    rect:pygame.Rect
    target:pygame.Rect
    label:str
    key:str
    affine:bool=False
    body:bool=True

class Renderer:
    def __init__(self,font_path):
        self.font_path=font_path;self.fonts={};self.state=None;self.scroll=0;self.max_scroll=0
        self.controls=[];self.text_runs=[];self.tooltips=[];self.bottom=0
    def font(self,role):
        sizes={'page':32,'section':24,'card':20,'body':18,'label':16,'small':15}
        key=(role,self.m.px(sizes.get(role,18)))
        if key not in self.fonts:self.fonts[key]=pygame.font.Font(self.font_path,key[1])
        return self.fonts[key]
    def wrap(self,text,width,role='body'):
        f=self.font(role);lines=[]
        for part in str(text).split('\n'):
            current=''
            for char in part:
                if current and f.size(current+char)[0]>max(1,width):lines.append(current.rstrip());current=char
                else:current+=char
            lines.append(current.rstrip())
        return lines or ['']
    def ellipsis(self,text,width,role='body'):
        f=self.font(role);text=str(text)
        if f.size(text)[0]<=width:return text
        if f.size('…')[0]>width:return ''
        low,high=0,len(text)
        while low<high:
            mid=(low+high+1)//2
            if f.size(text[:mid]+'…')[0]<=width:low=mid
            else:high=mid-1
        return text[:low]+'…'
    def text(self,rect,text,role='body',color=INK,lines=1,center=False):
        rect=pygame.Rect(rect);font=self.font(role);line_h=max(font.get_linesize(),font.get_height())
        parts=self.wrap(text,rect.w,role) if lines!=1 else [str(text)]
        limit=max(1,min(lines,rect.h//max(1,line_h)))
        if len(parts)>limit:parts=parts[:limit];parts[-1]=self.ellipsis(parts[-1]+'…',rect.w,role)
        if lines==1:parts=[self.ellipsis(str(text),rect.w,role)]
        if '\n'.join(parts)!=str(text):self.tooltips.append((rect.copy(),str(text)))
        for i,part in enumerate(parts):
            glyph=font.render(part,True,color)
            x=rect.centerx-glyph.get_width()//2 if center else rect.x
            y=rect.y+i*line_h+(max(0,(rect.h-glyph.get_height())//2) if lines==1 else 0)
            paint=glyph.get_rect(topleft=(x,y))
            clip=self.surface.get_clip();visible=paint.clip(clip)
            if visible.w and visible.h:
                self.surface.blit(glyph,(x,y))
                self.text_runs.append({'text':part,'role':role,'rect':list(visible),'container':list(rect),'overflow':paint.right>rect.right or paint.bottom>rect.bottom,'ellipsized':str(text)!=part and lines==1})
        return rect.bottom
    def body_rect(self,x,y,w,h):
        self.bottom=max(self.bottom,y+h)
        return pygame.Rect(x,y-self.scroll,w,h)
    def box(self,rect,color=WHITE):
        pygame.draw.rect(self.surface,color,rect,border_radius=self.m.px(6));pygame.draw.rect(self.surface,LINE,rect,1,border_radius=self.m.px(6))
    def register(self,rect,target,label,key=None,affine=False,body=True):
        if target is None:return
        self.controls.append(Control(pygame.Rect(rect),pygame.Rect(target),str(label),key or str(tuple(target)),affine,body))
    def button(self,rect,label,target=None,primary=False,enabled=True,key=None,selected=False,body=True):
        r=pygame.Rect(rect);color=BLUE if primary or selected else WHITE
        pygame.draw.rect(self.surface,color if enabled else (236,241,246),r,border_radius=self.m.px(5))
        pygame.draw.rect(self.surface,BLUE if selected else LINE,r,1,border_radius=self.m.px(5))
        self.text(r.inflate(-self.m.px(20),-self.m.px(6)),label,'body',WHITE if enabled and (primary or selected) else (INK if enabled else MUTED),center=True)
        if enabled:self.register(r,target,label,key,body=body)
    def choice(self,rect,label,checked,target,enabled=True,key=None):
        r=pygame.Rect(rect)
        if checked:pygame.draw.rect(self.surface,(231,241,249),r,border_radius=self.m.px(4))
        size=self.m.px(20);square=pygame.Rect(r.x+self.m.px(8),r.centery-size//2,size,size)
        pygame.draw.rect(self.surface,BLUE if checked else LINE,square,2,border_radius=3)
        if checked:pygame.draw.lines(self.surface,BLUE,False,[(square.x+4,square.centery),(square.centerx-1,square.bottom-4),(square.right-3,square.y+4)],2)
        self.text((square.right+self.m.px(10),r.y,r.w-size-self.m.px(24),r.h),label,'body',INK if enabled else MUTED)
        if enabled:self.register(r,target,label,key)
    def field(self,x,y,w,label,value,target,active=False):
        p=self.m.px;self.text(self.body_rect(x,y,w,p(25)),label,'label',MUTED)
        r=self.body_rect(x,y+p(29),w,p(44));self.box(r)
        if active:pygame.draw.rect(self.surface,BLUE,r,2,border_radius=5)
        self.text(r.inflate(-p(20),-p(4)),str(value)+('│' if active else ''),'body')
        self.register(r,target,label)
    def image(self,rect,image,label=''):
        r=pygame.Rect(rect);self.box(r);bitmap_rect=r
        if image is not None:
            try:
                if not isinstance(image,pygame.Surface):image=pygame.surfarray.make_surface(image.swapaxes(0,1))
                inner=r.inflate(-8,-8);scale=min(inner.w/image.get_width(),inner.h/image.get_height())
                size=(max(1,round(image.get_width()*scale)),max(1,round(image.get_height()*scale)))
                bitmap=pygame.transform.smoothscale(image,size);bitmap_rect=bitmap.get_rect(center=r.center);self.surface.blit(bitmap,bitmap_rect)
            except (TypeError,ValueError,AttributeError):image=None
        if image is None:self.text(r.inflate(-24,-24),label or '이미지 대기 중','label',MUTED,center=True)
        return bitmap_rect
    def heading(self,label,subtitle=''):
        c=self.m.content;p=self.m.px;y=c.y+p(20)
        self.text(self.body_rect(c.x,y,c.w,p(44)),label,'page')
        if subtitle:self.text(self.body_rect(c.x,y+p(52),c.w,p(48)),subtitle,'label',MUTED,2)
        return y+p(112 if subtitle else 76)
    def menu(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('실험을 시작할 준비','센서 설정과 데이터 수집부터 모델·학습 결과까지 순서대로 진행하세요.')
        left=int(c.w*.66) if c.w>p(940) else c.w
        items=[('센서 설정과 데이터 수집','타운과 카메라를 선택하고 실제 장면을 저장합니다.','data_gan_button'),('모델과 학습 설정','입력과 출력, 모델 구성과 학습 조건을 정합니다.','training_button'),('저장된 결과 확인','완료된 실험의 그래프와 예측을 확인합니다.','metric_button')]
        for idx,(label,hint,key) in enumerate(items):
            r=self.body_rect(c.x,y+idx*p(116),left,p(96));self.box(r)
            self.text((r.x+p(20),r.y+p(17),p(36),p(32)),str(idx+1),'section',BLUE)
            self.text((r.x+p(72),r.y+p(10),r.w-p(235),p(34)),label,'card')
            self.text((r.x+p(72),r.y+p(49),r.w-p(235),p(32)),hint,'label',MUTED)
            self.button((r.right-p(130),r.y+p(25),p(108),p(46)),'열기',g.get(key),key=key)
        if left<c.w:
            r=self.body_rect(c.x+left+p(24),y,c.w-left-p(24),p(328));self.box(r)
            self.text((r.x+p(20),r.y+p(18),r.w-p(40),p(34)),'현재 준비 상태','card')
            installed=(ROOT/'carla/CarlaUE4.sh').is_file()
            self.text((r.x+p(20),r.y+p(72),r.w-p(40),p(70)),'CARLA 0.9.15 준비됨' if installed else 'CARLA 설치 필요','body',TEAL,2)
            self.text((r.x+p(20),r.y+p(142),r.w-p(40),p(120)),'처음에는 Town01 · 전면 카메라 · 3장으로 시작하세요. 학습 전에 입력 파일을 확인합니다.','label',MUTED,5)
        self.text(self.body_rect(c.x,y+p(366),c.w,p(28)),'프로젝트 위치','label',MUTED)
        self.text(self.body_rect(c.x,y+p(401),c.w,p(38)),str(ROOT),'body')
    def configuration(self,g):
        from sdv.ui.widgets import checkbox_rect
        c=self.m.content;p=self.m.px;y=self.heading('타운과 센서 설정','한 번에 선택한 타운마다 같은 수집 수를 적용합니다. 카메라를 하나 이상 선택하세요.')
        tw=min(p(300),int(c.w*.3));sx=c.x+tw+p(24);sw=c.right-sx
        self.text(self.body_rect(c.x,y,tw,p(34)),'주행 타운','section')
        for i,town in enumerate(g.get('main_towns',[])):
            row,col=divmod(i,2);r=self.body_rect(c.x+col*tw//2,y+p(45)+row*p(48),tw//2-p(8),p(44))
            target=checkbox_rect(i//3,i%3,top_left=(80,150),cell_w=150,cell_h=120)
            self.choice(r,town,bool(g['town_checkboxes'].get(town)),target,key='town:'+town)
        self.choice(self.body_rect(c.x,y+p(250),tw,p(44)),'전체 타운',all(g['town_checkboxes'].values()),g.get('Town_all'))
        self.text(self.body_rect(sx,y,sw,p(34)),'수집 센서','section')
        name_w=sw-p(240)
        for j,label in enumerate(['분할','깊이','객체']):self.text(self.body_rect(sx+name_w+j*p(80),y+p(45),p(76),p(28)),label,'label',MUTED,center=True)
        for i,sensor in enumerate(g.get('sensors',[])):
            yy=y+p(80)+i*p(46);target=checkbox_rect(i,0,top_left=(580,150),cell_w=230)
            enabled=bool(g['sensor_checkboxes'].get(sensor));self.choice(self.body_rect(sx,yy,name_w,p(42)),LABELS.get(sensor,sensor),enabled,target,key='sensor:'+sensor)
            if sensor!='lidar':
                for j,opt in enumerate(['seg','depth','od']):
                    target=checkbox_rect(i,j+1,top_left=(580,150),cell_w=230)
                    self.choice(self.body_rect(sx+name_w+j*p(80),yy,p(76),p(42)),'',g['option_checkboxes'][sensor][opt],target,enabled,key='option:'+sensor+':'+opt)
        yy=y+p(464)
        for j,(key,label) in enumerate([('Cam_all','전체 카메라'),('seg_all','분할'),('Depth_all','깊이'),('od_all','객체')]):
            width=name_w if j==0 else p(80);xx=sx if j==0 else sx+name_w+(j-1)*p(80)
            cams=[v for v in g.get('cam_sensors',[]) if g['sensor_checkboxes'].get(v)]
            opt={'seg_all':'seg','Depth_all':'depth','od_all':'od'}.get(key)
            checked=all(g['sensor_checkboxes'].get(v) for v in g.get('cam_sensors',[])) if key=='Cam_all' else bool(cams) and all(g['option_checkboxes'][v][opt] for v in cams)
            self.choice(self.body_rect(xx,yy,width,p(44)),label,checked,g.get(key),key=key)
        active=g.get('train_modal_select_input')=='town_num';value=g.get('text_input','') if active else g.get('data_town_num',3)
        self.field(c.x,y+p(320),tw,'타운당 수집 수 (1~1000장)',value,g.get('data_num_rect'),active)
        count=sum(bool(v) for v in g['town_checkboxes'].values());total=int(g.get('data_town_num',3))*count
        self.text(self.body_rect(c.x,y+p(414),tw,p(74)),f'선택 {count}개 타운\n총 {total:,}장','body',TEAL,3)
    def parameters(self,g):
        from sdv.ui.widgets import checkbox_rect
        c=self.m.content;p=self.m.px;y=self.heading('학습 조건 설정','수치 입력 후 Enter로 적용하세요. 실행 전에 데이터와 가중치 준비 상태를 검사합니다.')
        gap=p(24);w=(c.w-2*gap)//3
        for i,(label,key,field) in enumerate([('학습률','input_LR','LR_rect'),('배치 크기','input_batch','batch_rect'),('에포크','input_Epochs','Epochs_rect')]):
            active=g.get('hyperparameter_input')=={'LR_rect':'LR','batch_rect':'batch','Epochs_rect':'Epochs'}[field]
            self.field(c.x+i*(w+gap),y,w,label,g.get('text_input','') if active else g.get(key),g.get(field),active)
        yy=y+p(116)
        for col,(title,keys,values,origin) in enumerate([('옵티마이저',g.get('opt',[]),g.get('opt_checkboxes',{}),(800,200)),('저장 옵션',g.get('save',[]),g.get('save_checkboxes',{}),(1100,200))]):
            xx=c.x+col*(w+gap);self.text(self.body_rect(xx,yy,w,p(34)),title,'section')
            for i,key in enumerate(keys):self.choice(self.body_rect(xx,yy+p(50)+i*p(52),w,p(44)),LABELS.get(key,key),values.get(key,False),checkbox_rect(i,0,top_left=origin,cell_h=50))
        self.text(self.body_rect(c.x+2*(w+gap),yy,w,p(34)),'추가 저장 형식','section')
        self.choice(self.body_rect(c.x+2*(w+gap),yy+p(50),w,p(44)),'DEEPX (미완성)',False,None,False)
    def model(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('입력·출력과 모델 구성','선택된 항목은 파랑으로 표시합니다. 수집 설정에 없는 입력은 사용할 수 없습니다.')
        gap=p(24);w=(c.w-2*gap)//3
        specs=[('전면','Front_cam_rect','Front_cam_color'),('왼쪽','Left_cam_rect','Left_cam_color'),('오른쪽','Right_cam_rect','Right_cam_color'),('후면','Rear_cam_rect','Rear_cam_color'),('후면 왼쪽','Rear_left_cam_rect','Rear_left_cam_color'),('후면 오른쪽','Rear_right_cam_rect','Rear_right_cam_color'),('라이다','lidar_rect','lidar_color')]
        self.text(self.body_rect(c.x,y,w,p(34)),'입력 센서','section')
        for i,(label,rect,color) in enumerate(specs):
            value=g.get(color);self.button(self.body_rect(c.x,y+p(48)+i*p(48),w,p(42)),label,g.get(rect),enabled=value!=g.get('GRAY'),selected=value==g.get('BLUE'))
        yy=y+p(404)
        for i,(label,key,value) in enumerate([('이미지 너비','width_rect','input_img_width'),('이미지 높이','height_rect','input_img_height')]):
            active=g.get('train_modal_select_input')==('width' if i==0 else 'height');self.field(c.x,yy+i*p(96),w,label,g.get('text_input','') if active else g.get(value),g.get(key),active)
        xx=c.x+w+gap;self.text(self.body_rect(xx,y,w,p(34)),'학습 출력','section')
        for i,(label,key,color) in enumerate([('BEV','bev_button_modal','bev_modal_color'),('이미지 분할','seg_button_modal','seg_modal_color'),('깊이','depth_button_modal','depth_modal_color'),('객체 검출','OD_button_modal','OD_modal_color')]):
            value=g.get(color);self.button(self.body_rect(xx,y+p(48)+i*p(52),w,p(44)),label,g.get(key),enabled=value!=g.get('GRAY'),selected=value==g.get('BLUE'))
        xx=c.x+2*(w+gap)
        groups=[('이미지 인코더',[('ResNet','rect_resnet_encoder','check_resnet_encoder'),('EfficientNet','rect_efficent_encoder','check_efficientnet_encoder')]),('깊이·분할 디코더',[('FPN','rect_resnet_decoder','check_fpn_decoder'),('Bi-FPN','rect_efficent_decoder','check_bifpn_decoder')])]
        if g.get('bev_modal_color')==g.get('BLUE'):groups.append(('BEV 디코더',[('ResNet','rect_resnet_bev_decoder','check_resnet_bev_decoder'),('EfficientNet','rect_efficent_bev_decoder','check_efficientnet_bev_decoder')]))
        for j,(title,items) in enumerate(groups):
            top=y+j*p(178);self.text(self.body_rect(xx,top,w,p(34)),title,'section')
            for i,(label,rect,key) in enumerate(items):self.choice(self.body_rect(xx,top+p(48)+i*p(48),w,p(44)),label,g.get(key,False),g.get(rect))
    def gallery(self,g,state):
        c=self.m.content;p=self.m.px;y=self.heading(TITLES.get(state,'데이터 확인'),'저장소 예시' if state in (1,6) else '실제로 수신하거나 저장된 이미지가 표시됩니다.')
        if state==1:
            summary='선택 타운: '+', '.join(g.get('selected_towns',[]))+'   ·   타운당 '+str(g.get('data_town_num',3))+'장'
            self.text(self.body_rect(c.x,y,c.w,p(42)),summary,'body',TEAL);y+=p(64)
        source='model_test_input_img' if state in (1,6) else ('model_data_gan_input_img' if state==3 else 'model_train_input_img')
        images=g.get(source,[]);items=[(i,image) for i,image in enumerate(images) if image is not None]
        if state==6:items.extend((i,image) for i,image in enumerate(g.get('model_test_output_img',[])) if image is not None)
        if not items:
            self.text(self.body_rect(c.x,y,c.w,p(90)),'이미지를 기다리고 있습니다.\n실행 상태와 오류 내용은 아래 상태 표시줄에서 확인하세요.','body',MUTED,3)
        if state==3:
            done=g.get('count_data_gan',0);total=int(g.get('data_town_num',3))*int(g.get('count',0))
            self.text(self.body_rect(c.x,y,c.w,p(36)),f'저장한 데이터 {done:,} / {total:,}장 · '+str(g.get('town','')).split('/')[-1],'body',TEAL);y+=p(56)
        cols=3 if c.w>=p(1320) else 2;w=(c.w-(cols-1)*p(20))//cols
        for j,(index,image) in enumerate(items):
            row,col=divmod(j,cols);r=self.body_rect(c.x+col*(w+p(20)),y+row*p(248),w,p(220));self.image(r,image)
            self.text(self.body_rect(r.x,y+row*p(248)+p(222),w,p(24)),f'이미지 {index+1}','small',MUTED)
        if state==3:
            yy=y+math.ceil(max(1,len(items))/cols)*p(248)+p(16)
            for i,(label,key) in enumerate([('RGB','RGB_button'),('분할','SEG_button'),('깊이','Depth_button'),('객체','OD_button')]):self.button(self.body_rect(c.x+i*p(140),yy,p(124),p(44)),label,g.get(key))
    def sensors(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('센서 위치와 카메라','숫자를 Enter로 적용한 뒤 적용 버튼으로 센서를 갱신하세요. 위치 그림과 실제 카메라를 구분해서 확인하세요.')
        names=list(g.get('selected_sensors',{}));idx=min(g.get('current_sensor_idx',0),max(0,len(names)-1));name=names[idx] if names else ''
        config=g.get('configs',{}).get(name,{});self.text(self.body_rect(c.x,y,c.w,p(32)),LABELS.get(name,name),'section');y+=p(46)
        left=int(c.w*.55);right=c.w-left-p(24);xx=c.x+left+p(24)
        for j,(image,key) in enumerate([(g.get('car_top'),'car_top_rect'),(g.get('car_side'),'car_side_rect')]):
            r=self.body_rect(c.x+j*(left+p(16))//2,y,(left-p(16))//2,p(216));bitmap=self.image(r,image)
            target=g.get(key)
            self.register(bitmap,target,'센서 위치 그림',key,affine=True)
            if target is not None and config:
                tx,ty=(g.get('px',target.centerx),g.get('py',target.centery)) if j==0 else (g.get('side_px',target.centerx),g.get('pz',target.centery))
                point=(bitmap.x+(tx-target.x)*bitmap.w/target.w,bitmap.y+(ty-target.y)*bitmap.h/target.h)
                clip=self.surface.get_clip();self.surface.set_clip(bitmap.clip(clip))
                pygame.draw.circle(self.surface,RED,(round(point[0]),round(point[1])),p(4))
                if 'lidar' not in name:
                    angle=-math.radians(config.get('yaw' if j==0 else 'pitch',0));direction=-1 if j==1 and 100<=(config.get('yaw',0)+360)%360<=260 else 1
                    for offset in (-20,20):
                        theta=angle+math.radians(offset)
                        end=(point[0]+direction*100*math.cos(theta)*bitmap.w/target.w,point[1]+(-1 if j==0 else 1)*direction*100*math.sin(theta)*bitmap.h/target.h)
                        pygame.draw.line(self.surface,RED,point,end,2)
                self.surface.set_clip(clip)
        self.image(self.body_rect(c.x,y+p(240),left,p(258)),g.get('sensor_img'),'실제 카메라 이미지 대기 중')
        labels={'x':'앞뒤 x (m)','y':'좌우 y (m)','z':'높이 z (m)','width':'너비 (px)','height':'높이 (px)','fov':'시야각 (°)','roll':'기울기 roll','pitch':'상하 pitch','yaw':'방향 yaw'}
        fields=[key for key in config if key not in {'seg','depth','od'}]
        for i,key in enumerate(fields):
            row,col=divmod(i,2);w=(right-p(16))//2;target=pygame.Rect(780+(i%3)*200,600+(i//3)*60,150,36)
            active=g.get('active_input')==(name,key);self.field(xx+col*(w+p(16)),y+row*p(96),w,labels.get(key,key),g.get('text_input','') if active else config[key],target,active)
        yy=y+math.ceil(len(fields)/2)*p(96)+p(10);self.button(self.body_rect(xx,yy,right,p(44)),'센서에 적용',g.get('set_button'))
    def queue(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('학습 대기열','설정을 저장한 뒤 입력 데이터와 실행 조건을 검사합니다.')
        items=g.get('Training_loop',[]);cols=3 if c.w>=p(1320) else 2 if c.w>=p(1000) else 1;w=(c.w-(cols-1)*p(20))//cols
        start=g.get('page',0)*g.get('buttons_per_page',6);items=items[start:start+g.get('buttons_per_page',6)]
        if not items:self.text(self.body_rect(c.x,y,c.w,p(82)),'대기열이 비어 있습니다. 추가 버튼으로 모델과 학습 조건을 준비하세요.','body',MUTED,3)
        for i,cfg in enumerate(items):
            row,col=divmod(i,cols);r=self.body_rect(c.x+col*(w+p(20)),y+row*p(302),w,p(282));self.box(r)
            self.text((r.x+p(20),r.y+p(15),r.w-p(120),p(36)),f'실험 {start+i+1}','card',BLUE)
            pairs=g.get('delete_buttons2',[]);target=pairs[i][0] if i<len(pairs) else None
            self.button((r.right-p(88),r.y+p(12),p(68),p(38)),'삭제',target,key='delete:'+str(i))
            task=', '.join(LABELS.get(k,k) for k,v in cfg.get('model_conf',{}).items() if v)
            sensor_use=cfg.get('sensor_use',{});model=cfg.get('model',{})
            lines=[f"에포크 {cfg.get('Epochs')}   배치 {cfg.get('batch')}",f"학습률 {cfg.get('LR')}",'출력: '+task,'입력: '+', '.join(LABELS.get(k,k) for k,v in sensor_use.items() if v),'인코더: '+str(model.get('encoder','')),'디코더: '+str(model.get('decoder',''))]
            for j,line in enumerate(lines):self.text((r.x+p(20),r.y+p(65+j*32),r.w-p(40),p(30)),line,'body')
        if g.get('total_pages',0)>1:
            yy=y+math.ceil(max(1,len(items))/cols)*p(302)+p(16)
            for i,(label,key) in enumerate([('이전 페이지','L_button'),('다음 페이지','R_button')]):self.button(self.body_rect(c.x+i*p(168),yy,p(152),p(44)),label,g.get(key))
    def results(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('저장된 결과','완료된 실험의 그래프·예측·지표를 확인합니다.')
        items=g.get('clickable_buttons',[])
        if not items and not g.get('delete_buttons'):
            r=self.body_rect(c.x,y,c.w,p(200));self.box(r)
            self.text((r.x+p(24),r.y+p(26),r.w-p(48),p(40)),'아직 저장된 학습 결과가 없습니다.','section')
            self.text((r.x+p(24),r.y+p(88),r.w-p(48),p(84)),'먼저 센서 설정과 학습 입력을 준비하세요. 중지한 실험이나 불완전한 결과는 완료된 결과로 표시하지 않습니다.','body',MUTED,3)
        for i,target in enumerate(items):
            label=PathLabel(g.get('Training_result',[]),i)
            self.button(self.body_rect(c.x,y+i*p(64),c.w,p(52)),label,target,key='result:'+str(i))
        yy=y+len(items)*p(64)
        for i,pair in enumerate(g.get('delete_buttons',[])):
            self.button(self.body_rect(c.x,yy+i*p(64),c.w,p(52)),str(pair[2])+' · 미완료 항목 삭제',pair[0],key='incomplete:'+str(i))
        yy+=len(g.get('delete_buttons',[]))*p(64)+p(16)
        if g.get('total_pages',0)>1:
            for i,(label,key) in enumerate([('이전 페이지','L_button'),('다음 페이지','R_button')]):self.button(self.body_rect(c.x+i*p(168),yy,p(152),p(44)),label,g.get(key))
    def inspection(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('결과 자세히 보기',str(g.get('selected_result','')))
        for i,(label,key) in enumerate([('그래프','metric_vis_graph_button'),('예측 이미지','metric_vis_img_button'),('지표','metric_vis_score_button')]):self.button(self.body_rect(c.x+i*p(172),y,p(156),p(44)),label,g.get(key))
        y+=p(72);image=g.get('img')
        if g.get('metric_vis_img_mode')=='score':
            for i,value in enumerate(g.get('score_list',[])):self.text(self.body_rect(c.x,y+i*p(32),c.w,p(30)),str(value),'body')
        else:self.image(self.body_rect(c.x,y,c.w,p(410)),image,'선택한 결과 파일을 확인하세요.')
        yy=y+p(438)
        for i,(label,key) in enumerate([('이전 결과','metric_L_button'),('다음 결과','metric_R_button')]):self.button(self.body_rect(c.x+i*p(172),yy,p(156),p(44)),label,g.get(key))
        slider=g.get('scrollbar_rect_vis')
        if slider is not None:
            r=self.body_rect(c.x,yy+p(72),c.w,p(24));pygame.draw.rect(self.surface,LINE,r,border_radius=8);self.register(r,slider,'예측 프레임',affine=True)
        if g.get('metric_vis_mode')=='BEV' and g.get('metric_vis_img_mode')=='graph':
            for i,target in enumerate(g.get('vis_bev_buttons',[])):self.button(self.body_rect(c.x+i*p(112),yy+p(124),p(104),p(42)),str(g.get('vis_bev',['손실','평균 IoU','도로','보행자','차량','차선'])[i]),target)
        for i,(label,key) in enumerate([('BEV','RGB_button'),('분할','SEG_button'),('깊이','Depth_button')]):self.button(self.body_rect(c.x+i*p(112),yy+p(184),p(104),p(42)),label,g.get(key),enabled=bool(g.get('selected_train_config',{}).get('model_conf',{}).get({'RGB_button':'BEV','SEG_button':'Seg','Depth_button':'Depth'}[key])))
    def running(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('학습 실행 상태','진행 수치는 실제 실행 로그가 있을 때 표시됩니다.')
        active=[j for j in OWNED if j.poll() is None and not j.stopped]
        message='첫 로그를 기다리고 있습니다.' if active else '실행 중인 학습 작업이 없습니다. 상태와 준비 파일을 확인하세요.'
        self.text(self.body_rect(c.x,y,c.w,p(72)),message,'section',MUTED,2)
        log=g.get('log')
        if isinstance(log,dict):
            message=f"에포크 {log.get('epoch','—')} / {g.get('input_Epochs','—')}   배치 {log.get('batchi','—')} / {log.get('last_idx','—')}   손실 {log.get('loss','—')}"
            self.text(self.body_rect(c.x,y+p(76),c.w,p(42)),message,'body',TEAL)
        yy=y+p(136)
        for j,job in enumerate(active):self.text(self.body_rect(c.x,yy+j*p(64),c.w,p(56)),f'{job.label} · {int(time.monotonic()-job.started)}초 / 90초\n로그: {job.log_path.name}','body',TEAL,2)
        images=[(i,image) for i,image in enumerate(g.get('model_train_input_img',[])) if image is not None]
        top=yy+p(104);cols=3 if c.w>=p(1320) else 2;w=(c.w-(cols-1)*p(20))//cols
        if not images:self.image(self.body_rect(c.x,top,c.w,p(240)),None,'실제 예측 출력 대기 중')
        for j,(index,image) in enumerate(images):
            row,col=divmod(j,cols);r=self.body_rect(c.x+col*(w+p(20)),top+row*p(244),w,p(216));self.image(r,image)
            self.text(self.body_rect(r.x,top+row*p(244)+p(216),w,p(24)),f'학습 이미지 {index+1}','small',MUTED)
        target=g.get('scrollbar_rect')
        if target is not None:
            bottom=top+max(1,math.ceil(len(images)/cols))*p(244)+p(16)
            r=self.body_rect(c.x,bottom,c.w,p(24));pygame.draw.rect(self.surface,LINE,r,border_radius=8);self.register(r,target,'예측 프레임',affine=True)
    def installation(self,g):
        c=self.m.content;p=self.m.px;y=self.heading('CARLA 설치','공식 기본 배포 7.81GiB · 중단한 파일은 다음 시도에서 이어받습니다.')
        self.text(self.body_rect(c.x,y,c.w,p(100)),g.get('install_message','설치 준비 중'),'body',MUTED,4)
        r=self.body_rect(c.x,y+p(124),c.w,p(12));pygame.draw.rect(self.surface,LINE,r,border_radius=6)
        progress=max(0,min(1,g.get('install_progress',0)));pygame.draw.rect(self.surface,TEAL,(r.x,r.y,int(r.w*progress),r.h),border_radius=6)
        for i,(label,target) in enumerate([('다시 시도',pygame.Rect(50,300,260,56)),('설치 취소',pygame.Rect(340,300,220,56))]):self.button(self.body_rect(c.x+i*p(188),y+p(176),p(172),p(48)),label,target)
    def toolbar(self,g,state):
        p=self.m.px;c=self.m.content;top=self.m.height-self.m.footer-self.m.action
        pygame.draw.rect(self.surface,WHITE,(0,top,self.m.width,self.m.action));pygame.draw.line(self.surface,LINE,(0,top),(self.m.width,top))
        if state==-1:self.button((c.right-p(112),top+p(12),p(112),p(44)),'종료',g.get('end_button'),body=False)
        elif state in (3,8):self.text((c.x,top+p(10),c.w,p(48)),'상단 중지를 누르면 이번 SDV 작업만 정리합니다. 저장된 파일은 남습니다.','label',MUTED,2)
        elif state in (9,10):self.button((c.right-p(116),top+p(12),p(116),p(44)),'홈',g.get('home2_button'),body=False)
        elif state==-2:return
        else:
            labels={0:'설정 검토',1:'센서 확인',2:'수집 시작',4:'모델 선택',5:'설정 검토',6:'대기열 추가',7:'준비 검사',11:'재학습 설정'}
            if state==2 and g.get('current_sensor_idx',0)<len(g.get('selected_sensors',{}))-1:labels[2]='다음 센서'
            self.button((c.right-p(156),top+p(12),p(156),p(44)),labels.get(state,'계속'),g.get('re_button') if state==11 else g.get('confirm_button'),primary=True,enabled=not self.shell.busy,body=False)
            self.button((c.x,top+p(12),p(100),p(44)),'뒤로',g.get('back_button'),enabled=state!=7,body=False)
            if state==7:self.button((c.x+p(116),top+p(12),p(116),p(44)),'실험 추가',g.get('loop_button'),body=False)
    def modal(self,g):
        if not (g.get('confirm_delete') or g.get('confirm_delete2')):return
        self.controls=[];p=self.m.px;r=pygame.Rect(0,0,min(self.m.content.w,p(520)),p(220));r.center=(self.m.width//2,self.m.height//2);self.box(r)
        self.text((r.x+p(24),r.y+p(24),r.w-p(48),p(42)),'저장된 항목을 삭제할까요?','section')
        self.text((r.x+p(24),r.y+p(82),r.w-p(48),p(44)),'취소하면 현재 항목을 유지합니다.','body',MUTED)
        self.button((r.x+p(24),r.bottom-p(70),p(140),p(44)),'취소',g.get('no_btn'),body=False)
        self.button((r.right-p(164),r.bottom-p(70),p(140),p(44)),'삭제',g.get('yes_btn'),body=False)
    def render(self,surface,g,state,shell,density=1):
        self.surface=surface;self.shell=shell;self.m=Metrics(*surface.get_size(),density);self.controls=[];self.text_runs=[];self.tooltips=[];self.bottom=self.m.content.y
        if state!=self.state:self.scroll=0;self.state=state;shell.focus=-1
        surface.set_clip(self.m.content)
        dispatch={-1:self.menu,-2:self.installation,0:self.configuration,2:self.sensors,4:self.parameters,5:self.model,7:self.queue,8:self.running,9:self.results,10:self.inspection}
        if state in (1,3,6):self.gallery(g,state)
        elif state==11:
            y=self.heading('새 학습 조건 준비','선택한 실험을 바탕으로 학습 조건을 다시 준비합니다.');self.button(self.body_rect(self.m.content.x,y,self.m.px(240),self.m.px(48)),'학습 조건 설정',g.get('re_button'))
        else:dispatch.get(state,self.menu)(g)
        self.max_scroll=max(0,self.bottom-self.m.content.bottom+self.m.px(24));self.scroll=max(0,min(self.scroll,self.max_scroll))
        surface.set_clip(None);self.toolbar(g,state);self.modal(g)
        if self.max_scroll:
            track=pygame.Rect(self.m.content.right+6,self.m.content.y+8,5,self.m.content.h-16);pygame.draw.rect(surface,LINE,track,border_radius=2)
            thumb_h=max(30,int(track.h*self.m.content.h/(self.m.content.h+self.max_scroll)));top=track.y+int((track.h-thumb_h)*self.scroll/max(1,self.max_scroll))
            pygame.draw.rect(surface,MUTED,(track.x,top,track.w,thumb_h),border_radius=2)
        shell.controls=[item.target for item in self.controls]
        if 0<=shell.focus<len(self.controls):
            focus=self.controls[shell.focus].rect.clip(self.m.content) if self.controls[shell.focus].body else self.controls[shell.focus].rect
            pygame.draw.rect(surface,TEAL,focus.inflate(4,4),2,border_radius=5)
    def target_at(self,pos):
        for control in reversed(self.controls):
            if control.rect.collidepoint(pos) and (not control.body or self.m.content.collidepoint(pos)):return control
        return None
    def logical(self,pos,control=None):
        control=control or self.target_at(pos)
        if control is None:return (-10000,-10000)
        if control.affine:
            return (control.target.x+(pos[0]-control.rect.x)*control.target.w/max(1,control.rect.w),control.target.y+(pos[1]-control.rect.y)*control.target.h/max(1,control.rect.h))
        return control.target.center

def PathLabel(paths,index):
    from pathlib import Path
    return Path(paths[index]).name if 0<=index<len(paths) else '저장된 실험'
