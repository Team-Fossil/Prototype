#include <Servo.h>
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;

unsigned long lastActionTime = 0;  // 마지막 액션 시간 기록

void setup() {
  servo1.attach(8);
  servo2.attach(9);
  servo3.attach(10);
  servo4.attach(11);
  servo5.attach(12);
  Serial.begin(9600);
  Serial.println("Ready to receive data...");
}

void loop() {
  if (Serial.available() > 0) {
    String receivedData = Serial.readStringUntil('\n');  // 줄바꿈 문자를 기준으로 데이터 읽기
    unsigned long currentTime = millis();  // 현재 시간 기록
    
    if (currentTime - lastActionTime >= 3000) {  // 3초가 지났는지 확인
      Serial.print("Received: ");
      
      if(receivedData == "ROCK"){
        Serial.print("OH! ");
        servo1.write(0);
        servo2.write(0);
        servo3.write(0);
        servo4.write(0);
        servo5.write(0);
      }
      else if(receivedData == "SCISSORS"){
        Serial.print("WOW! ");
        servo1.write(0);
        servo2.write(0);
        servo3.write(90);
        servo4.write(90);
        servo5.write(0);
      }
      else if(receivedData == "PAPER"){
        Serial.print("WAIT! ");
        servo1.write(90);
        servo2.write(90);
        servo3.write(90);
        servo4.write(90);
        servo5.write(90);
      }
      else if(receivedData == "UNKNOWN"){
        Serial.print("NO! ");
        servo1.write(90);
        servo2.write(90);
        servo3.write(90);
        servo4.write(0);
        servo5.write(0);
      }
      Serial.println(receivedData);
      
      lastActionTime = currentTime;  // 마지막 액션 시간 업데이트
    }
  }
}
