# 2024 D&A Conference
**AlphaCLIP + Pic2word**

<br/>

## 1. 배경 & 목적
 
- 기존의 Composed Image Retrieval(CIR) 은 전체 이미지와 텍스트를 통해 검색을 수행
- 이미지 속 객체에 더욱 집중하기 위해 지역 정보를 활용하고자 함
- 이를 통해, 기존의 모델들 보다 더욱 정교한 검색이 가능할 것이라고 판단

<br/>

## 2. 프로젝트 기간

- 팀 구성 : 2024년 7월 中
- 최종 발표 : 2024년 11월 15일

<br/>

## 3. Model FLow
### Model Architecture

![image](https://github.com/user-attachments/assets/d23a0e13-55e4-4bc1-8293-5477a8e963ed)

### Training flow

<img src = https://github.com/user-attachments/assets/549a6592-859d-46e5-a90b-41488295356d width=800>
![스크린샷 2024-12-06 135421]()




## 4. Model Experiments

- COCO Segmentation
  
| Method                     | R1    | R5    | R10   |
|----------------------------|-------|-------|-------|
| Pic2Word-Composed          | 11.5  | 24.8  | 33.4  |
| Pic2Word-Image             | 8.6   | 15.4  | 18.9  |
| Pic2Word-Text              | 6.1   | 15.7  | 23.5  |
| Pic2Word-Image + Text      | 10.2  | 20.2  | 26.6  |
| **Segment-Composed**       | 16.77 | **30.95** | **38.99** |
| Segment-Image              | 13.89 | 21.93 | 25.83 |
| Segment-Text               | 7.72  | 20.97 | 30.14 |
| **Segment-Image + Text**   | **17.23** | 28.75 | 35.13 |


- ImageNet-S
  
| Average                    | R1    | R5    | R10   | R50   | R100  | R200  |
|----------------------------|-------|-------|-------|-------|-------|-------|
| Pic2Word-Composed          | 1.41  | 6.71  | 10.75 | 23.36 | 30.27 | 37.86 |
| Pic2Word-Image             | 0.00  | 0.14  | 0.39  | 4.20  | 8.96  | 14.55 |
| Pic2Word-Text              | 0.14  | 0.36  | 0.49  | 2.04  | 4.19  | 8.07  |
| Pic2Word-Image + Text      | 0.00  | 0.72  | 1.63  | 11.38 | 19.75 | 28.14 |
| **Segment-Composed**       | **1.86** | **7.06** | **10.86** | **24.81** | **33.37** | **41.66** |
| Segment-Image              | 0.00  | 0.22  | 0.55  | 6.22  | 13.04 | 19.92 |
| Segment-Text               | 0.02  | 0.22  | 0.48  | 2.03  | 4.16  | 8.47  |
| **Segment-Image + Text**   | 0.10 | 3.49 | 6.49 | 22.52 | 31.44 | 38.92 |
