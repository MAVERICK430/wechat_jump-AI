import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import time

hwin = win32gui.FindWindow('Chrome_WidgetWin_0', '跳一跳')


def grab_screen(region=None):
    # print(hwin)
    # return

    width = region[2]
    height = region[3]
    left = region[0]
    top = region[1]

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


if __name__ == '__main__':
    img = grab_screen(region=(0, 0, 450, 844))
    print(win32api.GetKeyState(74) <= -127)
    time.sleep(1)
