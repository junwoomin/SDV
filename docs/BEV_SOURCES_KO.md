# BEV 소스 출처와 라이선스

[Global BEV 설명](GLOBAL_MAP_INTEGRATION_KO.md) · [출처 고지](../THIRD_PARTY_NOTICES)

## 코드 근거

| 자료 | 역할 |
| --- | --- |
| [LRS 좌표·지도 코드](https://github.com/junwoomin/LRS/tree/fe9cbc2f8f38fa5f7808e6d161e645160958c6a2) | `sim/utility.py`의 ego 정렬 crop 및 World/BEV 좌표 변환 근거 |
| [HBLR](https://github.com/junwoomin/HBLR/tree/7fcff7490fd97739d36d06d738aa9a3d422d6ad0) | BEV 지도·차선·다층 도로 문제에 대한 설명 자료 |
| [AGILEQ-Training 지도 코드](https://github.com/SungjinDavidLee/AGILEQ-Training/tree/f3eba609de1af9d07a1c3e37283afdc51e7fcf22/data_gen) | 사전 생성 PNG, ego 기준 추출과 높이 분기를 비교한 소스 |
| `sdv/bev/` | 명시적 좌표·유효성 정보, catalog 검사, 필요한 채널 로딩과 생성 구현 |

LRS에서 가져온 좌표 수식의 MIT 고지는 `sdv/bev/geometry.py`와 `THIRD_PARTY_NOTICES`에 유지합니다. Roach 정책의 별도 소스 정보는 `sdv/driving/vendor/roach/SOURCE.json`에 있습니다.

## 지도와 높이 자료

Global BEV는 별도 제공 지도 PNG와 일치하는 catalog를 사용합니다. 파일 이름이나 비교 이미지만으로 미공개 생성 알고리즘, 실제 차선 페인트 정답 또는 정밀 도로 높이를 확정할 수 없습니다. grayscale support, Semantic ID, metric elevation은 서로 다른 데이터 계약입니다.

원본 지도·체크포인트의 사용 및 재배포 권한은 코드와 별도로 확인해야 합니다. LRS·HBLR의 코드 MIT 고지를 AGILEQ custom 코드, 지도 이미지 또는 가중치 전체의 라이선스로 확대하지 않습니다. CARLA navigation 코드와 글꼴의 원래 고지는 해당 파일에 유지합니다.
