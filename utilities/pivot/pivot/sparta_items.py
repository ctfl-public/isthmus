# sparta_format.py

class SpartaItems:
    """This class is essentially acting as a data struct to hold the raw strings used in SPARTA style dump files. Should be updated if SPARTA style dumps are ever changed."""
    TIMESTEP = "ITEM: TIMESTEP"
    NUM_CELLS = "ITEM: NUMBER OF CELLS"
    BOX_BOUNDS = "ITEM: BOX BOUNDS"
    CELLS = "ITEM: CELLS"
    NUM_SURFS = "ITEM: NUMBER OF SURFS"
    SURFS = "ITEM: SURFS"
    
    XLO = "xlo"
    XHI = "xhi"
    YLO = "ylo"
    YHI = "yhi"
    ZLO = "zlo"
    ZHI = "zhi"
    
    REQUIRED_FLOW_FIELDS = {XLO, XHI, YLO, YHI, ZLO, ZHI}