import pyautogui
import time

print("Move your mouse to the top-left corner of the region and wait 5 seconds...")
time.sleep(5)
x1, y1 = pyautogui.position()
print(f"Top-left corner: ({x1}, {y1})")

print("Move your mouse to the bottom-right corner of the region and wait 5 seconds...")
time.sleep(5)
x2, y2 = pyautogui.position()
print(f"Bottom-right corner: ({x2}, {y2})")


"""
Pot: 483, 535
stack: 328, 736
board (?): 366, 412
street area (?): 490, 424
prev bet: 481, 230 
position (of the button): 423, 668
Card 1 (rank): 368, 411
Card 1 (suit): 370, 433
Card 2 (rank): 421, 409
Card 2 (suit): 423, 431
Card 3 (rank): 475, 407
Card 3 (suit): 476, 432
Card 4 (rank): 529, 410
Card 4 (suit): 530, 431
Card 5 (rank)
Card 5 (suit)
"""