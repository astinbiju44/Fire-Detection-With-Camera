import winsound

times=2
frequency=1000
duration=1000

def beepsound():
    for i in range(times):
        winsound.Beep(frequency,duration)
beepsound()