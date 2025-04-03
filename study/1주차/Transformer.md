# Attention is All You Need

## Transformer 모델
Transformer는 2017년 발표된 논문 “Attention Is All You Need”에서 처음 등장. 이름처럼 **오직 ‘어텐션’(attention) 메커니즘만 사용해서** 문장을 이해하고 생성하는 모델
기존 모델(RNN, LSTM 등)의 단점을 보완해서, **GPT, BERT, T5** 같은 거대한 언어모델들의 기초가 된 구조.

- **Self-Attention**: 입력 시퀀스의 각 단어가 다른 단어들과의 관계를 계산하여 표현을 재구성
- **Multi-Head Attention**: 다양한 관점에서 정보를 추출하고 병렬 처리
- **포지셔널 인코딩(Positional Encoding)**: 순서 정보가 없는 어텐션 구조에서 시퀀스의 위치 정보를 보완하기 위해 사용

## Transformer 목표
문장처럼 순서가 있는 데이터를 입력 받아서, 또 다른 순서 있는 데이터를 만들어내는 것(ex: 번역, 요약, 질문 생성 등)


## 모델 구조
[인코더 - 디코더 구조]
- **인코더**: 입력 문장 이해. self-attention + feed-forward 네트워크로 구성된 층 6개
- **디코더**: 이해한 정보 바탕으로 새로운 문장 생성. masked self-attention, encoder-decoder attention 등을 포함한 층 6개

`입력 문장 → [ 인코더 6층 ] → 중간 표현 → [ 디코더 6층 ] → 출력 문장`

![[스크린샷 2025-04-03 오후 3.03.28.png]]
### 인코더
- 입력 문장을 받아서 전체 문맥을 이해하는 역할

```css
입력 →
[ Multi-Head Self-Attention ] → 잔차 연결 + LayerNorm
→ [ Feed Forward Network (FFN) ] → 잔차 연결 + LayerNorm
→ 출력
```

- **Self-Attention**: 문장 내 단어들끼리 서로를 바라봄
- **FFN**: 각 단어를 독립적으로 처리하는 작은 신경망
- **LayerNorm & 잔차 연결**: 학습 안정화 + 정보 보존

### 디코더
- 인코더의 결과를 받아서 출력 문장을 하나씩 생성하는 역할

```scss
입력 →
[ Masked Multi-Head Self-Attention ]  → (앞 단어만 바라보기)
→ 잔차 연결 + LayerNorm
→ [ Encoder-Decoder Attention ]       → (입력 문장과 연결)
→ 잔차 연결 + LayerNorm
→ [ Feed Forward Network ]            → (단어 처리)
→ 잔차 연결 + LayerNorm
→ 출력
```

- **Masked Attention**: 출력 문장을 왼쪽 단어까지만 보고 다음 단어 생성 (auto-regressive 구조)
- **Encoder-Decoder Attention**: 인코더가 이해한 문맥을 참고해서 문장을 생성함

### 입력 처리 과정
1. 단어를 벡터로 임베딩
2. 각 단어에 **Positional Encoding** 추가 (순서를 반영하기 위해)
3. 이걸 인코더의 입력으로 사용

입력 = 임베딩 + 포지셔널 인코딩

## Self Attention
문장 안의 **각 단어가 다른 모든 단어를 바라보며**, 그 문맥(context)을 반영한 새로운 표현을 만드는 메커니즘

- RNN : 단어를 순서대로 처리 → 앞 단어의 정보만 봄
- Self-Attention: 단어 간 **모든 관계를 동시에** 봄 (병렬화 가능)

### 작동원리

#### Step 1: 단어 임베딩 → Q, K, V로 변환
- Q (Query): 내가 궁금한 것
- K (Key): 다른 단어들이 가진 정보의 ‘정체성’
- V (Value): 그 정보 자체

#### Step 2: 유사도 점수 계산

Query와 모든 Key 간의 내적(dot product)을 구함 
→ 이게 "얼마나 주목할지"를 의미함.

$$score = Q × Kᵗ$$

#### Step 3: 스케일링 및 소프트맥스

- dot product 값이 너무 크면 softmax가 뾰족해지니까 **스케일링** 해줌

$$score = (Q × Kᵗ) / √d_{k}$$

- Attention weight 는 문장 안에서 어떤 단어가 다른 단어를 얼마나 중요하게 생각하는지 나타내는 수치. →  각 Query 단어가 모든 Key 단어에 대해 얼마나 주목(attend)해야 하는지를 나타내는 확률 분포

$$ attention_{weights} = softmax(score)$$

### Step 4: Attention 값을 곱해서 최종 벡터 계산

$$output = attention_weights × V$$
#### 최종 식

$$Attention(Q,K,V)=softmax(Q×K^t/√d_{k}​)×V$$

- Q (Query), K (Key), V (Value): 단어 표현을 세 개로 쪼개서 계산
- √dk: softmax가 너무 뾰족해지는 걸 방지하는 **스케일링 요인**

## Multi-Head Attention
Self-Attention을 여러 개 동시에 병렬로 수행하는 방식이야. 다양한 시각에서 문장을 바라보게 해줌.

$$MultiHead(Q,K,V)=Concat(head1​,...,head8​)×Wo$$

## ## Feed-Forward Network (FFN)

각 단어 위치별로 독립적으로 처리하는 작은 MLP (2개의 Dense Layer). 어텐션이 문맥을 처리했다면, FFN은 개별 단어 정보를 강화하는 역할.



## Transformer 핵심 구성 요소

![[Pasted image 20250403171859.png]]
### 1. Word Embedding (단어 임베딩)

- 맨 아래에 `"let’s"`, `"to"`, `"go"`, `<EOS>`와 같은 단어들
- 이 단어들을 1-hot vector (파란색, 초록색, 빨간색 등)로 표현하고
- 그것을 고차원 공간의 **벡터(숫자)**로 바꿔주는 것이 **Word Embedding**

➡️ 결과적으로 `"go"`는 `[0, 0, 1, 0]` 같은 1-hot에서 **실수 벡터** `[1.09, 0.67]` 같은 숫자로 바뀜

### 2. Positional Encoding (위치 인코딩)
Transformer는 RNN처럼 순서를 하나씩 따르지 않기 때문에, **단어의 순서 정보가 없음.**  
그래서 **사인/코사인 기반의 수학적 방식으로 "순서 정보"를 추가**해야 해.

- 이미지에서는 각 단어 벡터에 파형처럼 생긴 "위치 벡터"를 더함 (`+`)
- 예: `[1.09, 0.67]` ← 단어 의미  
    `+ [위치 인코딩]` → 최종 벡터: `[합쳐진 값]`

→ 이렇게 하면 "go"가 문장 내 어디에 있는지도 표현 가능

![[스크린샷 2025-04-03 오후 5.23.53.png]]

### 3. Self-Attention (자기 어텐션)

**Transformer의 핵심 부분**
- 각 단어는 문장 내 **다른 모든 단어들을 바라보면서**, 자신이 누구와 어떤 관계가 있는지 판단함.
- 여기서 **Q (Query), K (Key), V (Value)** 세 가지로 변환하는 연산.

📌 이미지 중간의 초록 박스가 바로 이 부분이야:
- "Q", "K", "V"가 입력됨
- 그 결과, 각 단어가 다른 단어를 얼마나 주의할지 계산된 값이 나옴

예:
- "go"가 다른 단어들과 비교해서 나온 attention 결과: `[2.5, -2.1]`

➡️ 이 값이 softmax로 attention weight가 되고, 다시 Value에 곱해져서 문맥 반영된 벡터가 돼


### 4. Residual Connections (잔차 연결)
- Self-Attention을 통해 얻은 값에, **입력값을 그대로 더함(`+`)**
- 왜? → 학습이 **더 안정적**이고 **정보 손실을 방지**할 수 있음

예:
- Self-Attention 결과: `[2.5, -2.1]`
- 입력값: `[1.87, 1.09]`
- 결과: `[4.37, -1.01]` 처럼 더해짐 → 그리고 나서 **Layer Normalization**이 들어감

➡️ 이미지 위쪽의 `+` 박스들이 이 과정이야

| 단계                     | 설명                          |
| ---------------------- | --------------------------- |
| 1. Word Embedding      | 단어를 숫자 벡터로 변환               |
| 2. Positional Encoding | 단어의 순서 정보를 더함               |
| 3. Self-Attention      | 단어가 다른 단어들을 얼마나 주목할지 계산     |
| 4. Residual Connection | 계산된 벡터에 원래 입력을 더해서 정보 손실 방지 |

→ 병렬 연산이 가능해서 빠르게 학습