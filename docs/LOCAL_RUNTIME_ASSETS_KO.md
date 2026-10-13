# 로컬 실행 자산과 Git 반영 범위

현재 홈 PC에서는 기존 자산을 그대로 사용한다. Git 후보에는 실제 Studio/SDV
실행 코드, 설치·실행 스크립트, 테스트, 설정 예제, 문서와 UI 정적 파일을 포함한다.
원격에만 있는 기존 파일은 삭제하지 않았다. 이전 원격 README 전체도 보존했다.

다음 항목은 새로 Git에 추가하지 않는다.

- CARLA 배포와 설치 폴더, Python/Node 환경, 실행 상태·백업·로그
- 수집/학습/검증 데이터와 학습 결과·checkpoint·모델 가중치
- `roach/log/ckpt_11833344.pth`: 기존 로컬 Roach policy. 라이선스와 출처를
  확인한 자신의 checkpoint를 같은 경로에 놓거나 Settings에 프로젝트 안의 checkpoint 경로를 지정한다.
  없으면 CARLA 준비/수집 검사가 실제 누락 사유를 표시한다.
- `assets/bev_maps/data5`, `data10`: 기존 제공 Town PNG 지도를 프로젝트의
  같은 경로에 배치하거나 BEV Map Root를 프로젝트 안의 자신의 지도 경로로 지정한다.
  제공되지 않은 지도·높이를 임의 생성하지 않는다. 자산의 재배포 라이선스를
  코드의 MIT 라이선스로 대체하지 않으며 이 후보에는 지도 PNG를 넣지 않는다.
- 상세 validation JSON/log와 화면 캡처, 사용하지 않는 과거 Django archive

새 컴퓨터에서는 README의 환경 설치를 한 뒤 실제 CARLA 설치 위치와 연구
Python, 원본 dataset/calibration, map PNG, Roach checkpoint를 설정해야 한다.
Settings → System Diagnostics와 태스크 준비 검사를 통과한 뒤 실행한다.
기존 사용자 PC는 새 코드 적용 후 자산을 이동하거나 재설치할 필요가 없다.

원격에 이미 있던 소형 `img/test_img`와 과거 구현 파일은 원격 전용 내용을
보존하기 위해 그대로 유지했다. 새 수집 데이터나 가중치를 추가한 것은 아니다.
