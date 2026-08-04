##::::::::::::: Cavity Registry ::::::::::::::

# Optical Coefficients
PRMM3_COEFF = (1.031e-6)/(2**18)/4
PR2M3_COEFF = (1.031e-6)/(2**18)/4
PR2M2_COEFF = (2.243e-5)/(2**18)/4
PR2M1_COEFF = (1.563e-3)/(2**18)/2

SR2M3_COEFF = (1.031e-6)/(2**18)/4
SR2M2_COEFF = (2.588e-6)/(2**18)/4
SR2M1_COEFF = (1.563e-3)/(2**18)/2

BSM2_COEFF = (9.281e-5)/(2**18)/4
BSM1_COEFF = (1.234e-3)/(2**18)/2

M0_COEFF = (2.345e-4)/(2**18)/2
L1_COEFF = (5.254e-5)/(2**18)/4
L2_COEFF = (7.036e-7)/(2**18)/4
L3_COEFF = (5.285e-10)/(2**18)

IMCM3_COEFF = (1.031e-6)/(2**18)/4
IMCM2_COEFF = (2.588e-6)/(2**18)/4
IMCM1_COEFF = (1.563e-3)/(2**18)/2

FCM3_COEFF = (1.031e-6)/(2**18)/4
FCM2_COEFF = (2.588e-6)/(2**18)/4
FCM1_COEFF = (1.563e-3)/(2**18)/2

# :::::::::::: CPS Constructed Functions :::::::::::
#The following are hand built combinations using the corresponding CPS channels
# to represent the cavity's motion with CPS sensors

# ! These are not used in FIR engine, it is used in DELAY, and to calculate coherence
def DARM_CPS_Combo (cps):
  return 0.5*((cps['CPSEXDARM']-cps['CPSIXDARM'])-(cps['CPSEYDARM']-cps['CPSIYDARM']))

def PRCL_CPS_Combo (cps):
    H2, H3 = cps['CPSHAM2X'], cps['CPSHAM3X']
    BSX, BSY, IX, IY = cps['CPSBSX'], cps['CPSBSY'], cps['CPSIXCOM'], cps['CPSIYCOM']
    return 0.5*((H3-H2)+(H3-H2)+(BSX-H2)+(IX-BSX)+(IY-BSY))

def SRCL_CPS_Combo (cps):
    H4, H5 = cps['CPSHAM4Y'], cps['CPSHAM5Y']
    BSX, BSY, IX, IY = cps['CPSBSX'], cps['CPSBSY'], cps['CPSIXCOM'], cps['CPSIYCOM']
    return 0.5*((H5-H4)+(H5-H4)+(H5-BSY)+(BSX-IX)+(BSY-IY))

def MICH_CPS_Combo (cps):
    return (cps['CPSIXCOM']-cps['CPSBSX'])-(cps['CPSIYCOM']-cps['CPSBSY'])

def IMC_CPS_Combo (cps):
    return 0.5*(cps['CPSHAM3XIMC']-cps['CPSHAM2XIMC'])

def FC_CPS_Combo (cps):
    return cps['CPSHAM8Y']-cps['CPSHAM7Y']

# :::::::::::: Optical ::::::::::
# OPtical Targets calculated with suspension actuators

def DARM_OPT_Target (opt):
  return 0.5*((opt["ETMXM0"]*M0_COEFF + opt["ETMXL1"]*L1_COEFF + opt["ETMXL2"]*L2_COEFF)+(opt["ETMYM0"]*M0_COEFF + opt["ETMYL1"]*L1_COEFF + opt["ETMYL2"]*L2_COEFF) + 2*(opt["BSM1"]*BSM1_COEFF + opt["BSM2"]*BSM2_COEFF))

def PRCL_OPT_Target (opt):
  return (opt["PRM"]*PRMM3_COEFF + opt["PR2M3"]*PR2M3_COEFF + opt["PR2M2"]*PR2M2_COEFF + opt["PR2M1"]*PR2M1_COEFF)

def SRCL_OPT_Target (opt):
  return (opt["SR2M1"]*SR2M1_COEFF + opt["SR2M2"]*SR2M2_COEFF + opt["SR2M3"]*SR2M3_COEFF)

def MICH_OPT_Target (opt):
  return (opt["BSM1"]*BSM1_COEFF + opt["BSM2"]*BSM2_COEFF)

def IMC_OPT_Target (opt):
  return (opt["MC2M1"]*IMCM1_COEFF + opt["MC2M2"]*IMCM2_COEFF + opt["MC2M3"]*IMCM3_COEFF)

def FC_OPT_Target (opt):
  return (opt["FC2M1"]*FCM1_COEFF + opt["FC2M2"]*FCM2_COEFF + opt["FC2M3"]*FCM3_COEFF)



# :::::::::::: Cavity Registry ::::::::::
# A nested dictionary to hold the information

Cavity_Registry ={

'DARM': dict(CPS_Channels = ['CPSBSX','CPSBSY','CPSEXDARM','CPSEYDARM','CPSIYDARM','CPSIXDARM'],
            OPT_Channels = ['ETMXM0','ETMXL1','ETMXL2','ETMYM0','ETMYL1','ETMYL2','BSM1','BSM2'],
              target_func = DARM_OPT_Target,
            combo_func = DARM_CPS_Combo
            ),

'PRCL': dict(CPS_Channels = ['CPSHAM2X','CPSHAM3X','CPSBSX','CPSBSY','CPSIXCOM','CPSIYCOM'],
            OPT_Channels = ['PRM','PR2M3','PR2M2','PR2M1'],
              target_func = PRCL_OPT_Target,
            combo_func = PRCL_CPS_Combo
            ),

'SRCL': dict(CPS_Channels = ['CPSHAM4Y','CPSHAM5Y','CPSBSX','CPSBSY','CPSIXCOM','CPSIYCOM'],
            OPT_Channels = ['SR2M1','SR2M2','SR2M3'],
              target_func = SRCL_OPT_Target,
            combo_func = SRCL_CPS_Combo
            ),

'MICH': dict(CPS_Channels = ['CPSIXCOM','CPSIYCOM','CPSBSX','CPSBSY'],
            OPT_Channels = ['BSM1','BSM2'],
              target_func = MICH_OPT_Target,
            combo_func = MICH_CPS_Combo
            ),

'IMC':  dict(CPS_Channels = ['CPSHAM3XIMC','CPSHAM2XIMC','CPSEXDARM','CPSEYDARM','CPSIYDARM','CPSIXDARM'],
            OPT_Channels = ['MC2M1','MC2M2','MC2M3'],
              target_func = IMC_OPT_Target,
            combo_func = IMC_CPS_Combo
            ),

'FC':  dict(CPS_Channels = ['CPSHAM8Y','CPSHAM7Y','CPSEXDARM','CPSIXDARM','CPSEYDARM','CPSIYDARM'], #'CPSHAM7X', 'CPSHAM8X', 'CPSHAM7Z', 'CPSHAM8Z',
                            #'CPSHAM7RY', 'CPSHAM7RZ', 'CPSHAM8RX','CPSHAM8RY','CPSHAM8RZ',],
            OPT_Channels = ['FC2M1','FC2M2','FC2M3'],
              target_func = FC_OPT_Target,
            combo_func = FC_CPS_Combo
            )
}