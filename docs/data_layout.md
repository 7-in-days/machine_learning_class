# 데이터 폴더 규칙

저장소에는 실제 이미지 데이터를 포함하지 않는다. 학습이나 분석을 실행할 때는 별도 위치에 아래 구조를
준비하고 `--root` 또는 `HWATU_ROOT`로 경로를 지정한다.

```text
data/
  labels.csv
  raw/
    01/gwang/
    01/tti/
    01/pi/
    ...
    12/gwang/
    12/yeolggeut/
    12/tti/
    12/pi/
    no_card/
```

`labels.csv`는 다음 컬럼을 가져야 한다.

```csv
folder_path,card_id,month,type,type_id,description
01/gwang,01/gwang,1,gwang,1,1월 광
no_card,no_card,no_card,no_card,0,no card
```

같은 이미지가 month 모델과 type 모델에 모두 사용된다. `no_card` 이미지는 두 모델 모두에서
`no_card` 라벨로 학습한다.

