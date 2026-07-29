Flash the wireless display.txt into an esp32 connected to a JHD1602 display with connections mentioned in the script itself and wait for the display too show esp's IP address. Note this IP. 
Before running the Simulation_script_wo_radar.py  and Final_python_script.py , navigate through the code and find the part where IP address of esp is to be entered . enter the  IP address that we noted earlier, save and exit . 
For running the simulation_script_wo_radar.py , wait for dependencies to install and resolve and model weights to download .
when the system is ready , press enter to fire a speeding violation of 35kmph (which can be changed in the code itself )
final_python_script.py requires the AGD 307 radar to be connected via FTDI cable to operate . 