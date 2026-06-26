# ResNet 기반 화투패 실시간 분류 시스템

RealSense 카메라로 화투패를 실시간 촬영해 분류하는 학기 프로젝트다. 입력 이미지 한 장을 두 개의
ResNet18 모델에 동시에 넣고, 월(month)과 패 종류(type)를 각각 예측한다.

- month 모델: 1월~12월 + `no_card`, 총 13개 클래스
- type 모델: `gwang`, `yeolggeut`, `tti`, `pi`, `no_card`, 총 5개 클래스
- 최종 가중치: `checkpoints/month_final.pt`, `checkpoints/type_final.pt`

## 구조

```text
checkpoints/
  month_final.pt
  type_final.pt
src/hwatu/
  config.py
  data.py
  model.py
  analysis.py
scripts/
  train.py
  infer_realsense.py
  capture.py
  verify_dataset.py
  analyze.py
docs/
  data_layout.md
```

대용량 이미지, 실험 결과, 노트북, 중간 체크포인트는 저장소에 포함하지 않는다.

## 데이터

이미지는 `data/raw` 아래에 두고, `data/labels.csv`가 각 폴더의 라벨을 정의한다.

```text
data/
  labels.csv
  raw/
    01/gwang/
    01/tti/
    ...
    12/pi/
    no_card/
```

`labels.csv` 필수 컬럼은 다음과 같다.

```text
folder_path,card_id,month,type,type_id,description
```

경로는 `--root` 인자, `HWATU_ROOT` 환경변수, 저장소 루트 순서로 찾는다.

## 학습 설정

- backbone: `torchvision.models.resnet18(weights=None)`
- classifier head: Dropout 0.3 + Linear
- loss: CrossEntropyLoss
- optimizer: Adam, weight decay `1e-4`
- batch size: 64
- validation split: 20%
- best checkpoint: validation loss가 가장 낮은 epoch
- train sampler: `WeightedRandomSampler`, 샘플 weight는 `1 / class count`
- validation sampler: 사용하지 않음

학습 augmentation은 저장하지 않고 매 epoch 즉석에서 적용한다.

- 원본 40%
- 약한 augmentation 40%
- 강한 augmentation 20%

강한 augmentation은 `RandomRotation(180)`, `RandomPerspective(0.45, p=1.0)`,
`RandomAffine(translate=(0.16,0.16), scale=(0.7,1.25), shear=(-14,14))`, `ColorJitter`,
`GaussianBlur`를 사용한다. 좌우반전은 화투 그림 방향 때문에 사용하지 않는다.

## 실행

```bash
pip install -r requirements.txt

python scripts/verify_dataset.py --root /path/to/hwatu_project

python scripts/train.py --target month --root /path/to/hwatu_project
python scripts/train.py --target type --root /path/to/hwatu_project

python scripts/infer_realsense.py --demo

python scripts/analyze.py --target month --root /path/to/hwatu_project
python scripts/analyze.py --target type --root /path/to/hwatu_project
```

`--root`를 생략하면 `HWATU_ROOT`를 먼저 보고, 없으면 현재 저장소 루트의 `data/`를 사용한다.

## 최종 결과

최종 validation 결과는 보고서 기준 다음과 같다.

| 모델 | best epoch | val loss | val accuracy | correct / total |
|------|:---------:|:--------:|:------------:|:---------------:|
| type | 10 | 0.0071 | 0.9978 | 11,191 / 11,216 |
| month | 9 | 0.0043 | 0.9987 | 11,182 / 11,196 |

## 정리 내용

- `data_loader.py`, `modeling.py`, `result_analysis.py`를 `src/hwatu` 패키지로 정리
- 학습, 추론, 촬영, 검증, 분석 진입점을 `scripts/`로 분리
- RealSense 추론 스크립트 두 개를 `scripts/infer_realsense.py`로 통합
- RealSense 촬영 스크립트 두 개를 `scripts/capture.py`로 통합
- 하드코딩 절대경로를 CLI 인자, `HWATU_ROOT`, 저장소 상대경로 기반으로 교체
- 최종 체크포인트 두 개만 `month_final.pt`, `type_final.pt`로 보존

