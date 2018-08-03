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
				
				if 405 < y and y < 420 and x > 80:
					text_description.append(text["description"])

		if len(text_description) > 1:
			print(','.join(text_description))
					

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--json', type=str, help='JSON filename produced by detect_text.py')

	args = parser.parse_args()

	main(args)