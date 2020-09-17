# Flow Classification Method Using Variational Auto-Encoder in Software Defined Network Environment

System Development
- Python 3.7.6
- Pytorch 1.4.0
- 16GB RAM
- 3.30GHz Intel(R) i5-4590 CPU

ae_mlp.py
- Auto-Encoder와 MLP를 결합
- 먼저 Auto-Encoder를 플로우의 6가지 통계적 특성을 이용해 학습
- 학습된 Auto-Encoder에서 Encoder부분만 추출하여 MLP와 결합
- MLP를 학습시켜 MLP를 통해 플로우 분류

mlp.py
- 가장 기본적인 MLP를 이용한 플로우 분류 코드
- 32, 16 히든 레이어로 구성

vae_jsd.py
- 본 논문에서 제안하는 VAE를 통해 플로우의 잠재 변수의 분포를 추출하여 분포를 이용해 플로우 분류
- 먼저 플로우의 6가지 통계쩍 특성을 이용해 VAE를 학습
- 학습한 VAE에서 플로우의 잠재 변수의 분포 추출
- 추출한 분포를 JSD를 이용해 플로우 분류

vae_mlp(decoder&mlp).py
- VAE에서 Encoder에서 나온 잠재 변수를 디코더와 MLP의 인풋으로 활용
- 기존의 Reconstruction Error와 Regularization Error에 Clasification Error를 추가
- VAE에서 한번에 모든걸 학습

vae_mlp.py
- ae_mlp.py와 똑같이 VAE와 MLP를 결합

Reference
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
- Do, C. B. (2008). The multivariate gaussian distribution. Section Notes, Lecture on Machine Learning, CS, 229.
