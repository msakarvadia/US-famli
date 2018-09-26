import argparse
import os
import numpy as np
import json

def main(args):
	with open(args.json, "r") as f:
		text_desc = json.load(f)

		text_description=[args.json]
		for text in text_desc:
			if text["bounds"] and text["bounds"][0] and text["bounds"][0]["y"]:
				y = text["bounds"][0]["y"]
				x = text["bounds"][0]["x"]
				# {"x": 54, "y": 417}, {"x": 105, "y": 417}, {"x": 105, "y": 431}
				# [{"x": 87, "y": 418}, {"x": 109, "y": 418}, {"x": 109, "y": 429}, {"x": 87, "y": 429}]
				if args.miny < y and y < args.maxy and args.minx < x and x < args.maxx:
					text_description.append(text["description"])

		if len(text_description) > 1:
			print(','.join(text_description))
					

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--json', type=str, help='JSON filename produced by detect_text.py')
	
	parser.add_argument('--minx', type=int, help='x coordinate min', default=40)
	parser.add_argument('--maxx', type=int, help='x coordinate max', default=800)
	parser.add_argument('--miny', type=int, help='x coordinate min', default=405)
	parser.add_argument('--maxy', type=int, help='x coordinate max', default=420)
	parser.add_argument('--join', type=str, help='join character', default=',')

	args = parser.parse_args()

	main(args)