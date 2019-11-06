
from pynput import keyboard

class MyException(Exception): pass

def on_press(key):



# Collect events until released
with  as listener:
    try:
        listener.join()
    except MyException as e:
        print('{0} was pressed'.format(e.args[0]))
