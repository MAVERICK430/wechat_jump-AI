from ctypes import *
mouse = windll.LoadLibrary('./dll/DllDemo.dll')


mouse.click(1000)


