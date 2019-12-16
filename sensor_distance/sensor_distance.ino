int sensorPin = 2; // 적외선 센서 데이터 핀 선언
int i; // 모집군 순서
int num = 1; //모집군 수
int total = 0; //모집군 합
int average = 0; //모집군 평균
 
void setup() {
  
  Serial.begin(9600); // 시리얼 포트 개방
}
void loop() {
 
   for(i = 0;i <= num;i++) // 센싱 횟수 
   {
    float v = analogRead(sensorPin)*(5.0/1023.0); // 측정 전압 변환
    float di = 60.495*pow(v,-1.1904);  // 거리 계산
    total = total + di; 
  delay(10);
   }
  average = (int)total/num; // 평균
  if( i >= num){ // 초기화 
    i = 0;
    total = 0;
  }
 

 Serial.print("range : ");Serial.println(average); // 적외선 값 출력, -20은 시리얼 모니터로 보이는 내용과 실제 자에 표시된는 내용이 같도록 수정
}
