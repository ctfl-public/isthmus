import numpy as np
import sys

def progress_bar(c, total, message):
    """
    Displays a simple textual progress bar in the console.

    The bar updates every 10% of progress. Once the total is reached,
    it prints a newline and resets.

    Args:
        c (int): Current iteration index (0-based).
        total (int): Total number of iterations.
        message (str): Message to display alongside the progress bar.

    Returns:
        None: Prints progress bar directly to stdout.

    Notes:
        - Uses `progress_bar.c_decade` as a static attribute to track
          the next update threshold (default = 10).
        - The bar length is fixed to 10 characters (each "=" equals 10%).

    Example:
        >>> import time
        >>> for i in range(100):
        ...     progress_bar(i, 100, "Processing")
        ...     time.sleep(0.05)
        # Output:
        #     Processing... [=         ]
        #     Processing... [==        ]
        #     ...
        #     Processing... [==========]
    """
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


def noise_gen(mag):
    """
    Generates a random noise factor within ±(mag/100) of 1.
    """
    return (2*np.random.rand() - 1)*(mag/100) + 1