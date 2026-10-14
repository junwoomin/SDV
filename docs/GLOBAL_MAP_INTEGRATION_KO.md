# Global BEV 지도와 좌표

[환경 준비](SETUP_KO.md) · [소스 출처](BEV_SOURCES_KO.md)

## 지도 입력

기본 `global_raster` 경로는 별도로 준비한 Town PNG와 `catalog.json`을 읽습니다. catalog에는 실제 파일, SHA-256, World XY offset, 해상도와 채널 대응이 필요합니다. `assets/bev_maps/` 또는 프로젝트 내부의 Map Root를 설정에서 선택합니다. 저장소에 지도 패키지는 포함되어 있지 않습니다.

5 PPM은 0.2 m/pixel, 10 PPM은 0.1 m/pixel입니다. 지원 Town과 채널은 catalog 및 실제 파일 검사로 결정합니다. 원본 PNG의 alpha를 occupancy로 해석하지 않으며 실제 RGB support와 채널 계약을 사용합니다. 없는 high 채널을 low 이미지로 대체하지 않습니다.

## 좌표와 생성

`sdv/bev/global_map.py`가 선택한 Town·해상도의 파일 무결성을 확인하고 필요한 채널을 지연 로딩합니다. `sdv/bev/global_generator.py`는 World offset·해상도·ego pose를 사용해 로컬 BEV를 추출하고 동적 Actor, Route와 Traffic 정보를 결합합니다.

정적 지도, Actor Ground Z, 객체 Min/Max Z·Height와 지도 low/high 소속은 각각 다른 의미입니다. 높이 분기는 실제 지면 정보와 원본 지도 support를 함께 사용하며 선택 근거·누락 채널·경고를 메타데이터에 남깁니다.

## 높이와 유효성

low/high 지도 이름만으로 픽셀별 미터 도로 높이를 복원할 수 없습니다. 해당 높이 자료가 없으면 N/A와 유효성 마스크를 유지합니다. 겹친 도로와 경사로의 정확한 Layer 분류에는 실제 도로·지면 정합 검증이 필요합니다.

선택적 surface cache와 기본 PNG 경로는 서로 다른 라벨 출처입니다. BEV 스키마, 좌표·해상도와 수집 당시 보정을 맞춘 데이터셋만 함께 학습합니다. [데이터 파이프라인](PIPELINE_KO.md)과 [모델 범위](MODELS_KO.md)를 함께 확인하세요.
