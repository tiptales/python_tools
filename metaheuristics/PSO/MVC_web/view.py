
import sys
import compute

def get_input():
    """Get input data from the command line."""
    p = ' '.join(sys.argv[1:])
    return p


def present_output(p):
    """Write results to terminal window."""
    s = compute.compute(p)
    print(f'Similaritie: {s}.')