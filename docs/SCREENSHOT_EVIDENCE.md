# Real-screen evidence / 실제 화면 증거

Source revision: **junwoomin/SDV main `9fb7de2197e91575de818f15f7961ce3f4b05f75`**, inspected afresh on 2026-10-13. Every one of its 428 tracked files was byte-identical to the home `/home/wo/SDV` source before reorganization. Earlier references to `/home/wo/SDV/sdv_studio` describe prior validation locations; today's actual evidence identifies `/home/wo/SDV`.

The guide's 12 PNGs are copied byte-for-byte from original continuous browser frames, with no synthetic UI, AI edits, overlays, or fabricated numbers. [Machine-readable provenance](images/walkthrough/provenance.json) records clip/frame, SHA256, size and source revision for each. They show only the app content; no desktop notifications, account panels or browser chrome. Local installation paths are setup examples from the verified host. Application state warnings and genuine job notices, where present in the recording, are not edited to hide failures.

The original recording, 35 segments, timestamps and detailed `execution_evidence.json` remain on the owner PC. No new video is made or committed. Recording checksum: `8cb6384716d6f16a98a771e42c16b0702b4972aec0dbf157919fb5b7214cdd97`.

| Recorded fact / 기록 | Value / 값 |
| --- | --- |
| Collection job | `cf8de75bbb854ec6`, Completed, exit 0, 80/80 samples, 24.488 s |
| Dataset | `demo_20261013`, explicit scene selection, independent copy, source unchanged, no random frame mixing |
| Splits | Town01 TRAIN 40 / Town03 VAL 40 |
| Training job | `174f7591e7544c88`, Completed, exit 0, 3 epochs, global step 30, 30.215 s |
| Prediction records | 252 entries, 9 captures |
| Final VAL Seg mIoU | 0.0340709566 |
| Final VAL depth RMSE | 24.846230212 m |
| Final VAL BEV mean IoU | 0 |
| Final TRAIN / VAL total loss | 415.438684 / 13.644447; different scenes and task composition |
| Object annotations | TRAIN 10 positive frames / 10 boxes; VAL 0 positive frames / 0 boxes |

This is a functional demonstration, not convergence, generalization, driving-policy safety or detection-quality evidence. Precise BEV road elevation was not supplied; warnings, invalid masks and N/A remain. The screenshot's map layer `low` is a BEV layer identifier, separate from the engine's **Epic + Off-screen** quality profile.

사진은 같은 버전의 실제 저장 프레임이며, 수치는 짧은 기능 시연 기록입니다. VAL 객체 정답이 없고 BEV 지표도 낮으므로 성능으로 포장하지 않습니다. 정밀 도로 높이 누락을 임의로 채우지 않았습니다. 지도 레이어 `low`와 엔진 렌더링 품질 Low는 서로 다른 설정입니다.

## User-provided main dashboards / 사용자가 제공한 메인 화면

The README headers use two additional actual screenshots supplied by the user for GitHub publication: [English dashboard](images/dashboard-en.png) and [Korean dashboard](images/dashboard-ko.png). Their actual pixels confirm the corresponding language, Dashboard selection, 80 collected samples and the completed demo experiment. They are preserved byte-for-byte; no unexpected private information was visible. [Dashboard provenance](images/dashboard-provenance.json) records the original filenames, hashes and sizes.

두 README 상단은 사용자가 제공한 실제 한·영 대시보드를 각각 사용합니다. 이 두 사진은 기존 녹화 35개 구간에 포함된 사진이 아니라 새로 제공받은 메인 화면입니다. 총 사진은 단계 안내 12장과 메인 2장, 14장입니다. 화면의 완료 상태를 수렴 성능으로 표현하지 않습니다.
