## 인공지능을 이용한 금융 시계열 빅데이터 분석 및 예측 시스템 연구개발

<br>

> 팀원
 
백수민, 강현구, 김묘경, 서주현, 임예린, 조한별  <br><br>


> 프로젝트 기간
 
2021.09. - 2021.02. <br><br>


> 프로젝트의 필요성과 목적

인터넷과 각종 sns 등 다양하고 새로운 데이터 소스가 실시간으로 생겨나고 있다. 이를 빅데이터라 하는데
이러한 빅데이터 중 복잡계 시스템의 특성을 갖는 금융 시계열은 분석 및 예측이 가장 어려운 데이터이다.
복잡계 시계열은 과거 통계역학을 이용한 고전 적인 분석법과 현대의 최적화, 통계를 기반으로한 퀀트
분석법이 주를 이루고 있다. 하지만 이런 분석은 정적 인 정량분석에서만 효과적이고, 동적 환경에서의
실시간 분석, 예측은 그 정확도가 매우 낮은 단점이 있다. 이런 단점을 보완하기 위해 최근 들어 딥러닝 등
진보된 인공지능을 이용하여 복잡계 시계열 데이터를 분석하려는 연구가 많이 진행되고 있으며, 현재
변동성이 커가고 불확실성이 확대되고 있는 금융 시장에 적용하기 위한 노력 또한 여럿 시도되고 있다.
따라서 본 프로젝트에서는 복잡계 시스템에 적용하기 위한 빅데이터 유지보수 소프트웨어, 인공지능
연구를 진행하여 최대한 금융시장을 예측하는 방향으로 목적을 가지고 있다.<br><br>


> 실행환경

GOOGLE COLAB<br>
PYTHON 3.8

---


> 파일 설명

+ 📁 data_collection
  + KRX_fullcode_1.ipynb : KRX 상장된 주식 전종목 fullcode 수집 (초기 1회 실행)
  + KRX_fullcode_2.ipynb : KRX 주식 전종목 상태 (상장폐지, 신규상장 등) 매일 업데이트
  + datacollection_1.ipynb : KRX의 주식 전종목 날짜별 데이터 수집 (초기에 1회 실행, 약 6시간 소요)
  + datacollection_2.ipynb : KRX의 주식 전종목 날짜별 데이터 매일 업데이트 (하루에 1번 실행, 약 1시간 20분 소요)
 
+ 📁 data_preprocessing
  + AWS_preprocessing.ipynb : 클라우드 컴퓨팅을 이용한 대용량 주가데이터 병렬처리
  + preprocessing.py : AWS Scikit-Learn 전처리 스크립트
  + TimeSeries_trend.ipynb : 시가총액 trend [-1,1]값으로 변환
  + optimization.ipynb : 보조지표 파라미터 최적화 및 sell/buy signal 도출

+ 📁 modeling
  + final.ipynb : CNN, GRU 모델 구축, 파라미터 최적화 및 backtest 수익률 구현
  + modeling.ipynb : LSTM 사용하여 학습 및 예측
  + modeling_ML.ipynb : ElasticNet, RandomForest, XGBoost 사용하여 학습 및 예측
  + modeling_gru.ipynb : GRU 사용하여 학습 및 예측
