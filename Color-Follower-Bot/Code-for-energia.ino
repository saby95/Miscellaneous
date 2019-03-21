int l1=9;
int l2=8;
int r1=12;
int r2=13;

void setup()
{    
  pinMode(l1,OUTPUT);
  pinMode(l2,OUTPUT);
  pinMode(r1,OUTPUT);
  pinMode(r2,OUTPUT);
  Serial.begin(9600);
  Serial.write('3');
}

void loop() {
            char a;
  if(Serial.available()>0)
  {     
    a=Serial.read();
    switch(a)
    {
      case '2':
          digitalWrite(l1,HIGH);
          digitalWrite(l2,LOW);
          digitalWrite(r1,HIGH);
          digitalWrite(r2,LOW);
          break;

      case '3':
          digitalWrite(l1,LOW);
          digitalWrite(l2,LOW);
          digitalWrite(r1,LOW);
          digitalWrite(r2,LOW);  
          //delay(3500);
          break;
      case '1':
          digitalWrite(l1,LOW);
          digitalWrite(l2,HIGH);
          digitalWrite(r1,HIGH);
          digitalWrite(r2,LOW);
          break;
      case '0':
          digitalWrite(l1,HIGH);
          digitalWrite(l2,LOW);
          digitalWrite(r1,LOW);
          digitalWrite(r2,HIGH);
          break;
    }
    delay(500);
  }
}

