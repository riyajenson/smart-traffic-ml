#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define trig1 9
#define echo1 10
#define trig2 6
#define echo2 7

LiquidCrystal_I2C lcd(0x27, 16, 2);

long time1 = 0;
long time2 = 0;
float distanceBetween = 15.2; // CHANGE if different

void setup() {
  Serial.begin(9600);

  pinMode(trig1, OUTPUT);
  pinMode(echo1, INPUT);
  pinMode(trig2, OUTPUT);
  pinMode(echo2, INPUT);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Speed Meter");
  delay(2000);
  lcd.clear();
}

float getDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  return duration * 0.034 / 2;
}

void loop() {
  float d1 = getDistance(trig1, echo1);
  float d2 = getDistance(trig2, echo2);

  if (d1 < 15 && time1 == 0) {
    time1 = millis();
  }

  if (d2 < 15 && time1 != 0 && time2 == 0) {
    time2 = millis();
    float timeTaken = (time2 - time1) / 1000.0;
    float speed = distanceBetween / timeTaken;

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Speed:");
    lcd.setCursor(0, 1);
    lcd.print(speed);
    lcd.print(" cm/s");
    Serial.println(speed);   // one speed per line (cm/s)

    time1 = 0;
    time2 = 0;

    delay(1000);
  }
}
