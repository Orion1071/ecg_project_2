#include <Arduino.h>

int * arr_e = new int[1000]{};
int i_e = 0;
bool reset = false;
int counter = 0;
void setupT(){
    Serial.begin(115200);
}
void reseter(){
    arr_e = new int[1000]{};
    i_e = 0;
    counter = 0;

}
void loopT(){
    // if(Serial.available() && reset) {
    //     int * arr_e{new int[4000]{}};
    //     reset = false;
    // }
    while(Serial.available()){
        int tmp = (int) Serial.read();
        if(tmp != 0){
            arr_e[i_e] = tmp;
            i_e++;
        }
        
        
    }
    // reset = true;
    Serial.println("===============");
    Serial.println(sizeof(arr_e)/sizeof(arr_e[0]));

    for(int j = 0; j < i_e; j++) {
        Serial.printf("%d,",arr_e[j]);
    }
    Serial.printf("\ni = %d\n", i_e);
    Serial.println("===============");
    
    // counter ++;
    // if(counter == 10) {
    //     reseter();
    // }
    delay(1000);
}

/*
void loop(){
    Serial.println("i am working");
    while(Serial.available()){
        //int test
        // digitalWrite (LED_BUILTIN, HIGH);
        Serial.println("I have new data coming in");
        int length = 8;
        byte b[length];

        for(int a=length-1; a>-1;a--) {
            b[a] = Serial.read();
        }

        double* value = (double*)b;
        i= *value;
        serial_in[counter1] = *value;
        counter1 = (counter1+1)%(dim1*dim2);
        // digitalWrite (LED_BUILTIN, LOW);
    }
    Serial.printf("%d\n",i);
    Serial.printf("%d\n",counter1);


    if (counter1 >= 18809){
        digitalWrite (LED_BUILTIN, HIGH);
        for(int n = 0; n < dim1*dim2; n++) {
            Serial.printf("%d, ", serial_in[n]);
        }
    } else {
        digitalWrite (LED_BUILTIN, LOW);
    }
    delay(1);
}

*/