# 퍼스널 컬러 분류 알고리즘
## 간단한 프로젝트 설명  
- 본 프로젝트는 인공지능(AI)을 활용하여 보다 정확하고 경제적인 퍼스널 컬러 진단 솔루션을 제공함으로써, 개인 맞춤형 스타일링을 효율적으로 지원하는 것을 목표로 합니다
## 배경 및 목적 
- 최근 퍼스널 컬러 분석에 대한 관심이 증가함에 따라, 전문적인 진단 서비스를 제공하는 업체들이 늘어나고 있으나, 높은 비용으로 인해 많은 사람들이 부담을 느끼고 있습니다. 또한, 셀프 진단의 결과에 대한 신뢰성 부족과 재진단 수요가 빈번히 발생하고 있습니다. 본 프로젝트는 이러한 문제를 해결하기 위해 인공지능(AI)을 활용하여 보다 정확하고 경제적인 퍼스널 컬러 진단 솔루션을 제공하고, 이를 통해 개인 맞춤형 스타일링을 효율적으로 지원하는 것을 목표로 합니다.
## 데이터 라벨링 과정
1. 데이터 수집(Data Collection) : 퍼스널 컬러 분석을 위한 데이터셋은 Kaggle에서 제공하는 아시아 얼굴 이미지 데이터(10,000장)를 사용.[Kaggle 링크](https://www.kaggle.com/datasets/lukexng/aisanfaces)
2. 데이터 전처리(Data Preprocessing) : Dlib의 얼굴 탐지 모델을 활용하여 각 이미지에서 얼굴 영역을 식별하고, OpenCV를 이용해 빰과 턱의 피부 영역을 추출하였음. 추출된 피부 영역은 LAB 및 HSV 색상 공간으로 변환하여 각 영역의 평균값을 계산하였음.
3. 데이터 라벨링 : LAB 색상 공간의 B값 147을 기준으로 웜톤과 쿨톤을 분류하였고, HSV 색상 공간의 H 값 20및 S값 60을 기준으로 웜톤 중 봄과 가을으로 세분화하였고, 쿨톤의 경우 HSV의 S값 60과 LAB L값 70을 기준으로 쿨톤 중 여름과 겨울으로 분류하여 데이터 라벨링을 수행하였음.
4. <img width="670" alt="데이터셋 라벨링 과정" src="https://github.com/user-attachments/assets/75f0c7f7-943b-4bba-822d-e1360e306c92" /> (데이터 라벨링 과정)
## 모델의 구조
1. 모델 구조 : CNN(Convolutional Neural Network)을 기반으로 퍼스널 컬러를 4자기(봄 웜톤, 여름 쿨톤, 가을 웜톤, 겨울 쿨톤)으로 분류하는 딥러닝 모델을 설계함.
2. 학습 과정 : 데이터를 학습, 검증, 데스트 세트로 나누어 모델을 학습시켰으며, 교차 엔트로피 손실 함수와 Adam 옵티마이저를 사용하여 정확도를 최적화함.
3. 성능 평가 : (Evaluation) : 모델의 성능은 Accuracy를 주요 지표로 사용하였으며, Validation Accuracy를 통해 과적합 여부를 확인하였습니다.
4. <img width="469" alt="CNN모델 구조" src="https://github.com/user-attachments/assets/1f214153-4167-4727-b416-5645c6ad5d8b" /> (CNN모델 구조)
## 결과
1. 퍼스널 컬러 뷴류 결과
- 전체 테스트 데이터의 분포는 봄웜톤 : 30%, 여름 쿨톤 : 25%, 가을 웜톤 : 20%, 겨울 쿨톤 : 25%
- 각 분류 톤별 예측 정확도는 80~90% 범위를 기록하며, 겨울 쿨톤에서 가장 높은 성능을 보임.
2. 모델 학습 결과
- 초기 학습된 CNN모델은 Training Accuracy, Validation Accuracy 모두 높은 정확도를 보였으나 과적합이 발생함.
- 과적합 방지를 위해 Dropout, Batch Normalization을 추가 및 Learning Rate를 조정한 후, CNN 모델은 Training Accuracy 77%, Validation Accuracy 78%로 비교적 높은 정확도를 보였음.
- 이후, CNN모델을 학습할 때 Epoch 수를 늘려 90%에 가까운 높은 정확도의 값을 얻을 수 있었음.
## 연예인 테스트 결과
- 학습된 CNN 모델을 연예인 사진으로 테스트한 결과, 안정적으로 높은 성능을 보여주었으머, 퍼스널 컬러 분류에 대한 신뢰도를 입증할 수 있었음.
1. 봄웜톤<img width="391" alt="Suzy" src="https://github.com/user-attachments/assets/f651d354-1890-4585-8629-9e0cb7d5ec3e" />
2. 여름쿨톤 <img width="368" alt="Taeyeon" src="https://github.com/user-attachments/assets/d2223bb5-093b-46b8-b116-e7bfea3066db" />
3. 가을웜톤<img width="569" alt="Jennie" src="https://github.com/user-attachments/assets/eff36ab4-3db3-46fe-a39a-0110f8e5155d" />
4. 겨울쿨톤<img width="497" alt="Karina" src="https://github.com/user-attachments/assets/c22575d3-9fc1-45e2-a57e-b672dabc29c1" />
## 결론
- 본 연구를 통해 AI 기반의 퍼스널 컬러 진단 모델이 기존의 주관적이고 비용이 높은 전문 진단 서비스를 데체할 사능성을 보여줌
- CNN을 활용한 퍼스널 컬러 분류는 높은 정확도와 일관성을 보이며, 다양한 피부 톤 및 색상을 효과적으로 분석할 수 있음을 입증하였음.
- 연구 결과는 경제적이고 접근 가능한 맞춤형 스타일링 솔루션을 제공할 기반을 마련하며, 뷰티 산업 및 패션 관련 서비스의 혁신적인 도구로 활용될 가능성이 있음.
