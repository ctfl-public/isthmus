import numpy as np
import sys
#%% Miscellaneous Functions

def progress_bar(c, total, message):
    finished = 0
    if np.ceil(100*(c + 1)/total) == progress_bar.c_decade:
        sys.stdout.write('\r')
        finished = np.rint(10*(c + 1)/total).astype(int)
        sys.stdout.write('    ' + message + '... [' + '='*finished + ' '*(10 - finished) + ']')
        sys.stdout.flush()
        progress_bar.c_decade += 10
    if finished == 10:
        sys.stdout.write('\n')
        progress_bar.c_decade = 10
progress_bar.c_decade = 10

# generate a factor of +/- mag% to a number for testing 
def noise_gen(mag):
    return (2*np.random.rand() - 1)*(mag/100) + 1