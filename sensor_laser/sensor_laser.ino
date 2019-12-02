void setup ()
{
  Serial.begin(9600);
  pinMode (13, OUTPUT); // define the digital output interface 13 feet
}
void loop () {
   digitalWrite (13, HIGH); // open the laser head
   delay(1000);     // 1초 동안 딜레이.
}
