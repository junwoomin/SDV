# 학습 대기열·예측 이미지·태스크 지표

## 화면과 작업 흐름

학습 페이지는 **모델 학습 설정 / 학습 모니터링 / 학습 대기열**로 구성합니다. 실행 중에도 다음 모델의 설정과 학습·검증 경로를 지정하고 대기열에 추가할 수 있습니다. 각 작업은 추가 시점의 설정·센서 보정·저장 경로를 별도로 보관하며 이후 폼 저장으로 현재 작업의 설정이 바뀌지 않습니다.

대기열의 첫 작업부터 실행합니다. 같은 자원을 사용하는 연구 작업과 센서 미리보기는 기존 중복 실행 제한을 유지합니다. 준비 조건을 만족하지 못한 첫 작업은 대기 상태에서 실제 사유를 표시하고 다음 작업으로 임의로 건너뛰지 않습니다. 대기 작업의 위/아래 순서 변경은 대기열 전체 순서에 원자적으로 반영합니다.

| 상태 | 의미와 가능한 동작 |
|---|---|
| Queued | 실행 대기. 순서 변경·대기열 삭제 가능 |
| Running | 실제 연구 프로세스 실행 중. 일시정지 요청·중지 가능 |
| Pausing | 현재 Epoch의 학습·검증·예측·저장을 마치고 일시정지 준비 중 |
| Paused | 완료한 Epoch까지 저장한 상태. 다시 시작하면 대기열에 추가 |
| Completed | 전체 Epoch 완료 |
| Failed | 실제 실행 오류. 상세 로그와 마지막 완료 Checkpoint를 확인 |
| Stopped | 명시적 중지 또는 실행 전 대기열 삭제 |

대기열 삭제는 실행 예약을 제거하고 기존 실험 기록·설정·결과 파일을 보존합니다. 실행 중인 작업은 먼저 일시정지하거나 중지해야 삭제할 수 있습니다. **중지**는 현재 실행을 종료하는 동작이고, **일시정지**는 이어 학습할 상태를 저장하는 동작입니다.

SQLite에 작업 상태와 순서를 저장하므로 브라우저 새로고침으로 작업이 사라지지 않습니다. 프로세스의 PID와 생성 시각, 실제 이벤트와 종료 기록을 확인하며 종료 코드만으로 정상 일시정지했다고 판단하지 않습니다.

학습 모니터링은 실제 진행률·활성 경과 시간·남은 예상 시간·종료 예상 시각을 표시합니다. 완료한 Epoch의 실제 소요시간 평균으로 남은 시간을 추정하고 현재 진행한 시간을 차감합니다. Epoch 소요시간에는 검증·예측·Checkpoint 저장이 포함되며 일시정지 대기 시간은 포함하지 않습니다. 첫 Epoch가 완료되기 전에는 추정 근거가 없어 예상 시간을 준비 중으로 표시합니다. 예상 시간은 장치 부하·태스크·저장 성능에 따라 변하는 추정치입니다.

그래프 높이는 기존보다 약 1.5배 확대했습니다. 실행 로그는 기본으로 접혀 있으며 펼치면 실제 프로세스 출력을 확인할 수 있습니다. 실험 기록에는 인코더·태스크·Optimizer·Batch size·학습률을 함께 표시하고 **결과 확인**으로 상세 결과를 엽니다.

## 고정 표본 Prediction 기록

`sdv/training/predictions.py`는 실제 데이터셋의 RGB 프레임 목록에서 양 끝과 중간 위치를 포함해 최대 4개를 선택합니다. 데이터 내용으로 만든 fingerprint와 상대 프레임 ID를 사용하므로 동일한 데이터셋에서는 모델·태스크·배치 크기·Epoch가 달라도 같은 표본을 비교합니다. 표본이 4개 미만이면 실제 존재하는 표본만 사용하며 그 이유와 수를 표시합니다.

VAL은 기본적으로 매 Epoch 검증 후 고정 표본을 예측합니다. TRAIN은 기본적으로 전체 학습의 100 Step마다 고정한 최대 4개 학습 표본을 예측합니다. TRAIN 이미지도 별도 실제 inference 결과이며 임의 값이나 무작위 샘플 화면을 만들지 않습니다. Step은 옵티마이저 업데이트 수를 의미하며 재개 후 초기화하지 않습니다.

| 항목 | 기록 내용 |
|---|---|
| Segmentation | 입력 RGB, 클래스별 정답 색상 마스크, 실제 argmax Prediction 마스크 |
| Depth | 입력 RGB, 정답 깊이, 실제 예측 깊이. 동일한 0~100 m 표시 범위 |
| BEV | 입력 RGB와 DAS·Lane·Vehicle·Walker 각각의 정답/예측 이진 마스크 |
| Object Detection | 입력 RGB, 정답 Bounding Box, 실제 예측 Box·Class·Confidence |

선택한 카메라와 활성 태스크만 기록합니다. 예측은 `eval()`·`no_grad()`로 한 표본씩 실행하고 완료 후 모델·하위 모듈의 기존 training 상태와 Python/NumPy/Torch/CUDA RNG를 복원합니다. BatchNorm 통계나 다음 학습 배치의 난수 흐름을 예측 저장 때문에 바꾸지 않습니다. 실제 inference·이미지 저장 비용은 추가되므로 TRAIN 간격을 늘리거나 끌 수 있습니다.

이미지와 메타데이터는 다음 형식으로 누적합니다.

```text
<experiment-output>/predictions/index.json
<experiment-output>/predictions/<train|val>/epoch_XXXX/step_XXXXXXXX/capture_XXXXXX/
```

한 capture의 PNG 저장이 완료된 후에만 목록을 원자적으로 공개합니다. Epoch·Step·capture ID별 폴더로 보관해 재개 시 기존 이미지를 덮어쓰지 않습니다. 학습 화면과 결과 화면의 갤러리에서 분할·Epoch·표본·태스크를 선택하고 스크롤로 기록을 확인할 수 있습니다. 실험 비교는 동일한 VAL fingerprint·표본·태스크가 있는 모델들의 Epoch를 각각 선택해 이미지를 비교합니다. 이전 형식의 저장 이미지는 원래 파일을 읽되 없는 Epoch 정보를 임의로 부여하지 않습니다.

## 태스크 지표와 Best 표

`sdv/training/task_metrics.py`는 각 Epoch의 **전체 학습/검증 배치**에서 실제 예측과 정답을 누적합니다. 손실식·학습 경로는 유지합니다. 지표 정의와 최소/최대 방향은 Torch를 import하지 않는 `metric_registry.py`가 웹과 공유합니다.

| 태스크 | 주요 지표 | 좋은 방향과 집계 기준 |
|---|---|---|
| 전체 | Total loss | 낮을수록 좋음. 기존 배치 평균 Loss |
| Segmentation | mIoU, Pixel accuracy, Mean Dice, 클래스별 IoU | 높을수록 좋음. argmax 클래스와 전체 픽셀 confusion matrix |
| BEV | Mean IoU, DAS/Lane/Vehicle/Walker 각각의 IoU·Dice/F1·Precision·Recall | 높을수록 좋음. 독립 이진 마스크의 전체 TP/FP/FN 누적 |
| Depth | AbsRel, SqRel, RMSE, RMSE log, MAE | 낮을수록 좋음. 실제 유효 픽셀 수에 따른 가중 집계 |
| Depth | δ1, δ2, δ3 | 높을수록 좋음. 예측/정답 비율이 1.25, 1.25², 1.25³ 미만인 유효 픽셀 비율 |
| Detection | AP50, AP50:95, 클래스별 AP, Precision·Recall·F1 | 높을수록 좋음. 차량/보행자 클래스별 Box matching |

Segmentation의 union이 없는 클래스는 평균에서 제외하고 완벽한 IoU로 만들어 넣지 않습니다. 기존 `seg_iou`는 여러 클래스 logits를 독립적으로 0과 비교한 과거 지표이므로 새 multiclass `seg_miou`와 구분합니다. 기존 값은 이전 결과 호환을 위해 보존합니다.

BEV 예측은 logits > 0, 정답은 > 0.5로 판정합니다. Depth는 제공된 유효 마스크, 유한하며 0보다 크고 100 m 이하인 정답, 유한하고 양수인 예측을 사용합니다. 유효 픽셀 수·비정상 예측 수·예측 유효 비율도 함께 기록합니다. 계산할 표본이나 분모가 없는 지표는 임의로 0 또는 1로 채우지 않습니다.

Detection은 다음 정책으로 평가합니다.

- 원래 SDV Anchor와 회귀값을 실제 Box로 decode합니다.
- AP 후보 score >= 0.001, 클래스별 NMS IoU 0.5, NMS 전 클래스별 최대 1,000개 후보, 이미지별 최대 100개 예측을 사용합니다.
- IoU 0.50~0.95를 0.05 간격으로 평가하고 101개 recall 위치에서 보간 Precision을 평균합니다.
- Precision/Recall/F1은 IoU 0.50와 score >= 0.25에서 계산합니다.
- crowd·area별 분류 정보가 없는 SDV 데이터셋의 전체 영역 평가이며 완전한 COCO 평가 구현으로 부르지 않습니다.
- 갤러리 Box 표시는 가독성을 위해 score > 0.25를 사용하므로 AP 계산에 들어간 저신뢰도 후보가 모두 이미지에 그려지지는 않습니다.

Best 표는 지표 정의에 따라 최소값 또는 최대값과 **그 값이 나온 Epoch**를 표시합니다. 서로 다른 지표의 Best는 다른 Epoch에서 나올 수 있습니다. `best_loss_model.pth`는 최저 검증 Total loss 기준 Checkpoint이며 모든 지표의 Best를 동시에 담는 모델이라는 의미가 아닙니다.

실제 지표는 `train_metrics.csv`, `events.jsonl` 및 작업 기록에 저장합니다. 기존 후처리를 위한 과거 형식 CSV는 `legacy_train_metrics.csv`에 보존합니다.

## 일시정지와 이어 학습

엔진은 매 Epoch의 검증·예측·LR scheduler 업데이트가 끝나면 `resume_checkpoint.pth`를 임시 파일에 저장하고 파일 동기화 후 원자적으로 교체합니다. 저장 옵션에서 모델 전용 Checkpoint를 꺼도 재개 상태는 매 Epoch 보관합니다.

재개 Checkpoint에는 모델, 옵티마이저 모멘텀/상태, StepLR, 완료 Epoch·다음 Epoch·전체 Step, 최저 검증 Loss, 누적 지표, Python/NumPy/Torch CPU/CUDA RNG, 학습 설정과 데이터·센서 보정 signature가 들어 있습니다.

일시정지 요청은 현재 Epoch 중간 배치를 버리고 멈추지 않습니다. 예를 들어 Epoch 9 중에 요청하면 Epoch 9의 학습·검증·예측·저장을 완료하고 **Epoch 9까지 저장 / 재개는 Epoch 10부터** 상태가 됩니다. 마지막 Epoch에서 요청하면 전체 학습이 완료되므로 Completed로 처리합니다.

엔진은 `training_checkpoint`와 `training_paused` 이벤트를 내고 종료 코드 75로 종료합니다. 작업 관리자는 이 이벤트와 실제로 존재하는 소유 경로의 Checkpoint를 모두 확인해야 Paused로 기록합니다. 다시 시작하면 같은 실험 경로에 결과·로그·이벤트를 이어 기록합니다.

설정, 센서 보정 파일 내용, 실제 train/val 파일 내용·목록과 경로가 바뀌면 재개를 거부합니다. 크기나 수정 시각만 같아도 데이터 내용의 SHA256이 다르면 검출합니다. 임의의 모델 전용 `.pth`는 학습 재개 Checkpoint로 사용할 수 없습니다. 예기치 않은 종료에서도 직전 완료 Epoch의 원자적 파일은 보존하지만, 검증되지 않은 현재 배치가 저장되었다고 표시하지 않습니다.

작업 관리자는 제어 파일과 환경변수 `SDV_TRAIN_CONTROL`, `SDV_TRAIN_RESUME`를 전달합니다. 연구자가 별도 CLI로 재개 파일을 지정하는 경우 자신이 생성한 신뢰할 수 있는 Checkpoint만 사용해야 합니다.

## 모니터링 설정

학습 YAML의 기본값은 다음과 같습니다.

```yaml
monitor:
  train_preview_every: 100  # 0이면 TRAIN Prediction 저장 끄기
  val_preview_every: 1     # VAL Prediction 저장 간격, Epoch 단위
  preview_samples: 4       # 고정 표본 수, 현재 4개 계약
```

`train_preview_every`는 0~1,000,000의 정수, `val_preview_every`는 1~1,000의 정수입니다. 이전 YAML에 `monitor`가 없으면 기본값을 적용합니다. 기존의 하드코딩된 20 Step PNG 저장은 새 간격 설정과 중복되지 않도록 제거했으며 20 Step 텍스트 로그는 유지합니다.

## 검증 범위

실제 CPU Tensor를 사용해 표준 지표 계산, Anchor decode, 실제 SDV ResNet18/FPN 예측 PNG 생성, 고정 표본 선택과 RNG/모델 상태 보존을 검사했습니다. Adam·AdamW·SGD로 Epoch 1 완료 후 프로세스를 새로 구성한 뒤 재개했을 때 연속 실행과 Loss·가중치·LR scheduler·RNG가 정확히 일치하는 것을 확인했습니다. 저장 실패 시 이전 완료 Checkpoint가 남는지도 확인했습니다.

전체 연구 엔진에 대해서도 테스트에서만 `.cuda()` 전송과 CUDA 필수 검사를 CPU로 대체해 실제 Dataset·ResNet18/FPN·Seg/Depth·손실·Adam·StepLR·지표·예측·Checkpoint 경로를 실행했습니다. 생성한 64×64 PNG 표본 각 4개로 2 Epoch 연속 실행과 Epoch 1 일시정지/프로세스 종료/Epoch 2 재개를 비교했고 가중치·전체 태스크 지표·이력이 정확히 일치했습니다. 고정 VAL 4개, 누적 Prediction 48개, TRAIN 간격 0 비활성화도 확인했습니다. 이는 합성 QA 표본의 코드 연결 검증이며 연구 데이터의 인지 성능 측정은 아닙니다. CPU 엔진 검증 기록 (local-only artifact: `training-engine-cpu-validation.json`; not distributed / 로컬 검증 자료·미배포)에 명시적인 CPU 대체 범위를 기록했습니다.

이 검사에서 기존 collater가 Depth 유효 마스크를 float32로 바꾸어 원래 `silog_loss`의 Tensor 인덱싱이 실패하는 오류를 발견했습니다. 마스크의 bool 자료형을 보존하도록 수정했고 실제 원래 Loss의 forward/backward 회귀 검사를 추가했습니다. Loss 공식은 변경하지 않았습니다.

재현은 Torch·Torchvision·EfficientNet PyTorch·NumPy·Pillow·OpenCV·SciPy·Pandas·Matplotlib·PyQuaternion·PyYAML이 설치된 연구 Python에서 다음 명령으로 실행합니다. CPU 대체는 명시적으로 허용해야 하며 일반 학습 실행에는 적용되지 않습니다.

```bash
python scripts/check_training_monitor_cpu.py --allow-test-cpu-adaptation
```

스크립트는 프로젝트 `.runtime`의 별도 임시 폴더에만 합성 데이터·설정·Checkpoint를 만들고 성공 후 삭제합니다. 확인용 데이터를 남기려면 `--keep-output`을 지정합니다. `--report`로 별도 검증 기록 경로를 지정할 수 있습니다.

웹/작업 대기열 경계 검증과 실제 CUDA 장치에서 모든 모델·태스크·데이터로 전체 학습을 실행한 검증은 구분합니다. **실제 CARLA 수집·CUDA 전체 학습·GPU 장치에서 pause/resume은 현재 미검증**입니다. 이번 변경에 대한 최종 실행 테스트 및 브라우저 검증 결과는 [검증 문서](VALIDATION_KO.md)를 확인하세요.
