# Universal Chemistry and Thermodynamics Constants (SI units)
M_CO2                    = 44.01e-3  # Molar mass of Carbon Dioxide (kg/mol)
M_AIR                    = 28.97e-3  # Molar mass of dry atmospheric air (kg/mol)
M_H2O                    = 18.015e-3 # Molar mass of water (kg/mol)
CP_AIR                   = 1005.0    # Specific heat capacity of dry air (J/kg·K)
R_IDEAL_GAS              = 8.314     # Ideal gas constant (J/mol·K)
R_DRY_AIR                = R_IDEAL_GAS / M_AIR # 287.05  # Specific gas constant for dry air (J/kg·K)
P_ATM_STANDARD           = 101_325.0  # Standard reference atmospheric pressure (Pa)
RHO_AIR_STANDARD         = 1.204
CP_LIQ_WATER             = 4186.0   # J/kg·K
CP_ICE                   = 2050.0    # Specific heat capacity of water ice (J/kg·K)
RHO_ICE_KG_M3             = 917.0     # Ice near 0 C, rounded from USACE EM 1110-2-1612; only for attachment-film mass
FACE_BOND_REFREEZE_TEMP_C = -0.1      # Attachment melt credit resets after the inner cell cools below this
INV_CP_ICE               = 1.0 / CP_ICE
H_FUSION                 = (51059.0 - 45051.0) / M_H2O   # Murphy & Koop 2005 Table C1 at 273.15 K: L_ice - L_liq, J/mol -> J/kg (= 333.5 kJ/kg)

MOLAR_RATIO_H2O_AIR      = M_H2O / M_AIR  # Ratio of water vapor to dry air molecular weight (~0.622)
MOISTURE_BUOYANCY_FACTOR = (1 / MOLAR_RATIO_H2O_AIR) - 1 # 0.608 constant used in gas constant adjustment

ALDUCHOV_LIQ_CONST_SET = (610.94, 17.625, 243.04)
ALDUCHOV_ICE_CONST_SET = (611.21, 22.587, 273.86)

# Harrison (1963) Latent Heat of Vaporization Constants
HARRISON_VAP_ENTHALPY_0C = 2501.0
HARRISON_VAP_TEMP_COEFF = 2.361

# Murphy & Koop (2005) Constants for Ice Sublimation
MK_SUB_ENTHALPY_INTERCEPT = 46782.5
MK_SUB_ENTHALPY_LINEAR_COEFF = 35.8925
MK_SUB_ENTHALPY_QUAD_COEFF = 0.07414
MK_SUB_LOW_T_SCALE = 541.5
MK_SUB_LOW_T_REF = 123.75
MK_SUB_INV_LOW_T_REF = 1 / MK_SUB_LOW_T_REF

# --- Frost layer (Phase 3) ---
# da Silva, Hermes & Melo (2011), Appl. Therm. Eng. 31, 2616-2621, Eq. 9 (fan-supplied tube-fin evaporators)
FROST_DENSITY_PREFACTOR_KG_M3 = 494.0       # density scale
FROST_DENSITY_T_FS_COEFF_PER_C = 0.11       # sensitivity to frost-surface temperature
FROST_DENSITY_T_DEW_COEFF_PER_C = -0.06     # sensitivity to dew point of the air at the coil inlet
FROST_DENSITY_T_FS_MIN_C = -10.0            # lowest coil temperature tested (Table 1); density is not extrapolated below it
# Lee, Kim & Lee (1997), Int. J. Heat Mass Transfer 40, 4359-4365, Eq. 15
FROST_K_INTERCEPT_W_M_K = 0.132             # conductivity extrapolated to zero density
FROST_K_LINEAR_COEFF = 3.13e-4              # per (kg/m3)
FROST_K_QUADRATIC_COEFF = 1.6e-7            # per (kg/m3)^2
FROST_K_DENSITY_MIN_KG_M3 = 50.0            # stated validity range of the conductivity fit
FROST_K_DENSITY_MAX_KG_M3 = 400.0
# Coil pressure drop ~ Q^2 / A_min^n. n = 2 calibrated against da Silva (2011) reported blockage/airflow loss
# (approximate readings, range 1.7-2.4); the ideal derivation gives 3.
FROST_BLOCKAGE_PRESSURE_EXPONENT = 2.0
FREE_FLOW_FRACTION_FLOOR = 0.02             # numerical guard when fins close

# Defrost controller phases
DF_NORMAL        = 0
DF_WAIT_FOR_STOP = 1
DF_SETTLE        = 2
DF_HEAT          = 3
DF_DRIP          = 4
DF_FAN_DELAY     = 5

# Defrost event termination codes (zero means no termination decision yet).
DF_TERMINATION_PROBE   = 1
DF_TERMINATION_TIMEOUT = 2


# cfm -> m3/s : 1 cfm = 0.000471947 m3/s
CFM_TO_M3S = 0.000471947

CELSIUS_TO_KELVIN = 273.15
KELVIN_TO_CELSIUS = -273.15

# Reciprocals
INV_3600 = 1.0 / 3600.0
INV_M_CO2 = 1.0 / M_CO2
INV_CP_AIR = 1.0 / CP_AIR
INV_H_FUSION = 1.0 / H_FUSION

# Sentinels and safe floors
SAFE_PRESSURE_THRESHOLD = 10_000.0
SAFE_FLOOR = 1e-30
MASS_KG_FLOOR = 1e-9 # NOTE: DEBUG
SENTINEL_FLOAT64 = -999.0
SENTINEL_INT64 = -999
SENTINEL_FLOAT8 = -99.0
SENTINEL_INT8 = -99
# SENTINEL_NEG_ONE = -1.0
MAX_EPOCH_TIMESTAMP_MS = 253402300799000
MIN_EPOCH_TIMESTAMP_MS = -62135596800000

MS_PER_HOUR = 3_600_000
MS_PER_DAY = MS_PER_HOUR * 24
MS_PER_12_HOURS = MS_PER_HOUR * 12
MS_PER_24_HOURS = MS_PER_DAY
MS_PER_48_HOURS = MS_PER_HOUR * 48
MS_PER_72_HOURS = MS_PER_HOUR * 72
MS_PER_5_MINUTES = int(5 * 60 * 1e3)
MS_PER_10_MINUTES = 2 * MS_PER_5_MINUTES

EPOCH_MS_TOKEN_HASTINGS    = -28400000000000 # Year 1066 (Battle of Hastings)
EPOCH_MS_TOKEN_MAGNA_CARTA = -24200000000000 # Year 1215 (Magna Carta)
EPOCH_MS_TOKEN_AGINCOURT   = -17500000000000 # Year 1415 (Battle of Agincourt)
