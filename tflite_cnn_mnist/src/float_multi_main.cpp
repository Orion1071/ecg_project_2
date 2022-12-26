#include <Arduino.h>
#include <stdio.h>
#include <string.h>

// const int numBytes = 4;
// const int dim1 = 28;
// const int dim2 = 28;
// float arr[dim1 * dim2];

/*
float multiply(float arr[]){
  float mul = 0.0;

  for(int i = 0; i < dim1 * dim2; i++) 
    mul = mul * arr[i];

  return (mul);
}
 
void setup(){
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);
    Serial.begin(115200);

}

void loop(){
    
    int curr_dim = 0;
    byte array[numBytes];

    while (Serial.available() == 0)
    {
      //send '!' up the chain
      Serial.println('!');

      //spam the host until they respond :)
      delay(10);
    }

    while (curr_dim < dim1 * dim2) {
      // turn on the LED to indicate we're waiting on data
      digitalWrite(LED_BUILTIN, HIGH);

      // wait until we have enough bytes
      while (Serial.available() < numBytes) {}

      for (int i = numBytes -1 ; i > -1; i--)
      {
        array[i] = Serial.read();
      }

      // print out what we received to just double check
      Serial.print("Byte array received was: 0x");
      for (int i = 0; i < numBytes; i++)
      {
        Serial.print(array[i], HEX);
      }
      Serial.println("");

          // now cast the 32 bits into something we want...
      float value = *((float*)(array));
      arr[curr_dim] = value;
      
      // print out received value
      Serial.print("Value on my system is: ");
      Serial.printf("%f\n",arr[curr_dim]);
      curr_dim ++;
      digitalWrite(LED_BUILTIN, LOW);

      Serial.println('$');
      delay(20);
    }

    Serial.print("The sum of the received array is: ");
    float mul = 0.0;
    for(int f = 0; f < dim1 * dim2; f++) {
      mul = mul + arr[f];
    }
    // Serial.printf("%f %f %f %f\n",arr[0],arr[1],arr[2],arr[3] );
    Serial.printf("%f\n", mul);
    Serial.println("%");
    curr_dim = 0;


}

*/