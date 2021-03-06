"""
simple helper functions to create mappings for the column names of different
brain atlases and regions
"""

import pandas as pd
from typing import Union


def colnames_to_yeo_7(colnames: list, order: bool = True) -> list:
	"""
	takes a list of colnames in the brainnetome format/naming and converts them to yeo_7 regions

	Examples:
	>>> print(colnames_to_yeo_7(["108_110", "1_2", "200_218", "148_140"]))
	>>> print(colnames_to_yeo_7(["108_110", "1_2", "200_218", "148_140"], order=False))
	Args:
		colnames: list of the brainnetome colnames
		order: whether the resulting colnames should be ordered

	Returns:
		list of yeo_7 converted colnames
	"""

	lookup = {
		1: 6, 2: 4, 3: 7, 4: 6, 5: 7, 6: 7, 7: 3, 8: 3, 9: 2, 10: 2, 
		11: 7, 12: 6, 13: 7, 14: 7, 15: 4, 16: 6, 17: 6, 18: 6, 19: 6, 
		20: 6, 21: 6, 22: 6, 23: 7, 24: 6, 25: 3, 26: 3, 27: 5, 28: 6, 
		29: 6, 30: 3, 31: 6, 32: 6, 33: 7, 34: 7, 35: 7, 36: 6, 37: 4, 
		38: 4, 39: 4, 40: 4, 41: 7, 42: 7, 43: 7, 44: 7, 45: 5, 46: 6, 
		47: 5, 48: 5, 49: 5, 50: 5, 51: 7, 52: 7, 53: 2, 54: 2, 55: 3, 
		56: 3, 57: 2, 58: 2, 59: 2, 60: 2, 61: 4, 62: 4, 63: 3, 64: 3, 
		65: 4, 66: 2, 67: 2, 68: 2, 69: 5, 70: 5, 71: 2, 72: 2, 73: 2, 
		74: 2, 75: 2, 76: 2, 77: 5, 78: 5, 79: 7, 80: 7, 81: 7, 82: 6, 
		83: 7, 84: 7, 85: 3, 86: 3, 87: 7, 88: 7, 89: 5, 90: 5, 91: 3, 
		92: 3, 93: 5, 94: 5, 95: 7, 96: 5, 97: 3, 98: 3, 99: 6, 100: 6, 
		101: 5, 102: 5, 103: 5, 104: 5, 105: 1, 106: 1, 107: 3, 108: 1, 
		109: 5, 110: 5, 111: 5, 112: 1, 113: 1, 114: 1, 115: 5, 116: 5, 
		117: 5, 118: 5, 119: 1, 120: 1, 121: 7, 122: 7, 123: 4, 124: 4, 
		125: 3, 126: 3, 127: 3, 128: 3, 129: 3, 130: 3, 131: 2, 132: 2, 
		133: 3, 134: 3, 135: 1, 136: 1, 137: 6, 138: 6, 139: 3, 140: 3, 
		141: 7, 142: 6, 143: 3, 144: 7, 145: 2, 146: 2, 147: 6, 148: 6, 
		149: 2, 150: 3, 151: 1, 152: 1, 153: 7, 154: 7, 155: 2, 156: 2, 
		157: 2, 158: 2, 159: 3, 160: 2, 161: 2, 162: 2, 163: 2, 164: 2, 
		165: 0, 166: 6, 167: 4, 168: 4, 169: 4, 170: 4, 171: 2, 172: 2, 
		173: 4, 174: 4, 175: 7, 176: 7, 177: 0, 178: 0, 179: 7, 180: 4, 
		181: 7, 182: 1, 183: 4, 184: 4, 185: 4, 186: 4, 187: 7, 188: 7, 
		189: 1, 190: 1, 191: 1, 192: 1, 193: 1, 194: 1, 195: 1, 196: 1, 
		197: 1, 198: 1, 199: 1, 200: 1, 201: 3, 202: 1, 203: 1, 204: 1, 
		205: 1, 206: 1, 207: 1, 208: 1, 209: 1, 210: 1, 211: 0, 212: 0, 
		213: 0, 214: 0, 215: 0, 216: 0, 217: 0, 218: 0, 219: 0, 220: 0, 
		221: 0, 222: 0, 223: 0, 224: 0, 225: 0, 226: 0, 227: 0, 228: 0, 
		229: 0, 230: 0, 231: 0, 232: 0, 233: 0, 234: 0, 235: 0, 236: 0, 
		237: 0, 238: 0, 239: 0, 240: 0, 241: 0, 242: 0, 243: 0, 244: 0, 
		245: 0, 246: 0
		}
	
	splitted = [[int(j) for j in i.split("_")] for i in colnames]
	new_names = [sorted([lookup[i] for i in j]) if order else [lookup[i] for i in j] for j in splitted]
	return [str(i[0]) + "_" + str(i[1]) for i in new_names]


def get_colnames_df(df: bool = True) -> Union[pd.DataFrame, dict]:
	"""
	generates a DataFrame or Dictionary which maps from the brainnetome (246x246)
	connectivity column names to the yeo_7 network column names

	Examples:
	>>> print(get_colnames_df())
	>>> print(get_colnames_df(False))
	Args:
		df: whether to return the column name mappings as a DataFrame

	Returns:
		a DataFrame or dictionary containing the column name mappings
	"""
	conn_names = [str(i) + "_" + str(j) for i in range(1, 247) for j in range(1, 247)]

	return (pd.DataFrame({
		"conn_name": conn_names, 
		"region": colnames_to_yeo_7(conn_names)
		}) if df 
		else dict(zip(conn_names, 
			colnames_to_yeo_7(conn_names))))


if __name__ == "__main__":
	print(colnames_to_yeo_7(["108_110", "1_2", "200_218", "148_140"]))
	print(colnames_to_yeo_7(["108_110", "1_2", "200_218", "148_140"], order=False))
	print(get_colnames_df(True))
	
