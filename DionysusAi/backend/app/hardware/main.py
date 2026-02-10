import machine
import time

#GPIO32 (D32)
adc = machine.ADC(machine.Pin(32))
adc.atten(adc.ATTN_11DB)

print("=== MQ-3 Alcohol Sensor ===")
print("Preheating 10-20 min...")

while True:
    raw = adc.read()
    voltage = raw * 3.3 / 4095
    print(f"VOLTAGE:{voltage:.3f}") 
    time.sleep(1)