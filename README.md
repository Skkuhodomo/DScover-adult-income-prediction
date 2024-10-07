# 2024 2학기 DScover 가이드 프로젝트
2024-2  성균관대학교 데이터분석 학회 DScover 가이드 프로젝트 E조

## 인구 소득 예측 프로젝트(Adult Census Income Prediction)
데이콘에서 진행한 **인구 데이터 기반 소득 예측 AI 해커톤**을 진행하였습니다.

[대회 페이지로 이동](https://dacon.io/competitions/official/235892/overview/description)



## Quick Start
## MacOS, WindowsOS, LinuxOS

레포지토리 복제
```shell
git clone https://github.com/Skkuhodomo/DScover-adult-income-prediction
```

라이브러리 설치
```shell
pip install -r requirements.txt
```


케글에서 데이터셋을 다운로드 합니다. 그 후, './data/'위치로 이동시킵니다.

[데이터셋](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

---

## Data Analysis
세부 내역은 `main.ipynb`을 참고하길 바란다. 

### 데이터 정의

```txt 
id : 샘플 아이디
age : 나이
workclass : 일 유형
fnlwgt : CPS(Current Population Survey) 가중치
education : 교육수준
education.num : 교육수준 번호
marital.status : 결혼 상태
occupation : 직업
relationship : 가족관계
race : 인종
sex : 성별
capital.gain : 자본 이익
capital.loss : 자본 손실
hours.per.week : 주당 근무시간
native.country : 본 국적
target : 소득 
```
---

## Model 
###  LightGBM 
Tree  구조를 기반으로 한 gradient boosting framework이다. Leaf-wise (Best-first) Tree Growth 구조를 택하기 때문에 Level-wise Tree Growth 구조를 사용하는 XGBoost보다 빠른 속도로 학습할 수 있다. 
![lightGBM2](https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png)
![lightGBM](https://lightgbm.readthedocs.io/en/latest/_images/level-wise.png)



###  TabNet 
본 프로젝트의 진행 과정에서 주관이 반영된 Feature Engineering의 한계를 느껴, Attention-layer를 사용한 딥러닝 모델인 TabNet을 통한 임베딩을 진행, 그 후 분석을 진행하였다. 해당 코드는 `tabnet_embedding.ipynb`를 참고하면 된다. 

![Tabnet](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*PCyFIs8ce-a5j4caAhJiVg.png)
위 그림은 TabeNet의 구조를 나타낸다. 

결과적으로 딥러닝을 통한 개선을 크게 보지 못하였다. 그 이유는, 데이터가 25000개로 비교적 적다는 점, 데이터 자체의 이상치가 존재할 가능성을 배제하지 못하기 때문이다. 

$$
\text{cosine similarity} = \frac{A \cdot B}{\|A\| \|B\|}
$$


정확도가 떨어지는 원인 분석을 위해 (데이터 누수에 해당하지만) 임베딩 후 코사인 유사도를 기반으로 train과 test의 유사한 행들을 그룹화하였다. 그 결과 범주형 열의 차이가 income의 결과를 다르게 만드는 경우, 즉 다른 데이터는 모두 동일한 경우, 범주형의 차이가 실제 값과 다르게 예측하게 하는 원인으로 파악하였다. 

## Insight
본 프로젝트를 통하여 다음과 같은 인사이트를 도출할 수 있었다. 
[데이터 분석을 통한 인사이트 추출](./data_analysis_insight.md)
[딥러닝을 통한 인사이트 추출](./deep_learning_insight.md)

## Contributors

| 이름 | 전공 | 깃허브 주소 |
| --- |--|  --- |
| 박호진(**팀장**) |시스템경영공학/데이터사이언스융합학과| |
| 심세윤 |문헌정보학/소프트웨어학| |
| 한석호 |시스템경영공학/차세대반도체공학 | |
